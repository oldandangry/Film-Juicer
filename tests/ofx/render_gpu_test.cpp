
#include "render_gpu_fixture.h"
#include "render_assertions.h"
#include "../ffi/case_results.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <cuda.h>
#include <nlohmann/json.hpp>
#include <cstdio>
#include <exception>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
    constexpr std::string_view kCaller = "ofx.gpu.render";
    constexpr auto kRoutes = std::span(ofx_test::kRenderReferenceCases).first<4>();

    constexpr std::array kBoostRows{"highlight/negative-direct", "highlight/negative-print-enlarger", "highlight/positive-direct-camera", "highlight/positive-print-composed"};

    OfxPlugin* candidate() {
        return OfxGetPlugin(0);
    }
    using Json = nlohmann::json;
    Json words(const std::vector<float>& values) {
        auto result = Json::array();
        for (const auto value : values)
            result.push_back(std::bit_cast<std::uint32_t>(value));
        return result;
    }
    void route_case(const ofx_test::RenderReferenceCase& route, const ofx_test::RenderPixels& expected) {
        ofx_test::DeviceImages images({7, 5, route.components});
        ofx_test::RenderHost fixture(candidate());
        ofx_test::configure_reference_render(fixture, images, route);
        const bool print = std::string_view(route.route) == "NegativePrintScan";
        if (print) {
            fixture.host().set_property(fixture.host().action_args(), kOfxImageEffectPropRenderWindow, {route.origin.x, route.origin.y, route.origin.x + 6, route.origin.y + 5});
            fixture.host().clear_calls();
            ofx_test::require_render(fixture.render() == kOfxStatErrFatal, "print route accepted partial frame");
            for (float sample : images.read())
                RenderAssertions::require_same_bits("print partial frame destination", sample, ofx_test::DeviceImages::kCanary);
            const auto& calls = fixture.host().calls();
            ofx_test::require_render(std::none_of(calls.begin(), calls.end(), [](const auto& call) {
                                         return call.operation == "abort" || (call.operation == "get_double" && call.property == kOfxImageEffectPropFrameRate);
                                     }) &&
                                         fixture.host().image_release_count(fixture.effect(), "Source") == 1 && fixture.host().image_release_count(fixture.effect(), "Output") == 1,
                                     "print partial frame reached later work or retained images");
            fixture.host().set_property(fixture.host().action_args(), kOfxImageEffectPropRenderWindow, {route.origin.x, route.origin.y, route.origin.x + 7, route.origin.y + 5});
        }
        ofx_test::require_render(fixture.render() == kOfxStatOK, "route render failed");
        const auto actual = images.pixels(images.read());
        for (float value : actual)
            ofx_test::require_render(std::isfinite(value) && value >= 0.0f && value <= 1.0f, "route pixel outside finite output range");
        const auto& predicted = expected.at(route.id);
        RenderAssertions::compare_pixels(route.id, actual, predicted);
        if (route.components == 4)
            for (std::size_t alpha = 3; alpha < actual.size(); alpha += 4)
                RenderAssertions::require_same_bits("route alpha", actual[alpha], predicted[alpha]);
        if (std::string_view(route.id) == "route/negative-print") {
            auto altered = predicted;
            altered.front() += 1.0F;
            bool rejected = false;
            try {
                RenderAssertions::compare_pixels("candidate altered reference", actual, altered);
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            ofx_test::require_render(rejected, "candidate accepted altered print reference");
            RenderAssertions::compare_pixels(route.id, actual, predicted);
        }
        const std::size_t releases = print ? 2 : 1;
        ofx_test::require_render(fixture.host().image_release_count(fixture.effect(), "Source") == releases &&
                                     fixture.host().image_release_count(fixture.effect(), "Output") == releases,
                                 "route image release count");
    }
    void highlight_case(std::size_t index) {
        constexpr int width = 33;
        constexpr int height = 17;
        constexpr int x = -11;
        constexpr int y = 7;
        const auto& route = kRoutes[index];
        ofx_test::DeviceImages images({width, height, 4});
        ofx_test::RenderHost fixture(candidate());
        fixture.configure(images, {x, y});
        const auto set = [&](const char* name, std::vector<ofx_test::InstanceValue> values) {
            fixture.host().set_value(fixture.effect(), name, std::move(values));
        };
        const auto change_boost = [&](double stops) {
            set("HalationHighlightBoostEv", {stops});
            auto* args = fixture.host().change_args(
                {kOfxChangeUserEdited, kOfxTypeParameter, "HalationHighlightBoostEv"}, 37.0);
            ofx_test::require_render(fixture.call(kOfxActionInstanceChanged, fixture.effect(), args) == kOfxStatReplyDefault,
                                     "highlight boost edit");
        };
        set("FilmProfileKey", {std::string(route.film)});
        set("ScanRoute", {std::string(route.route)});
        set("InputCctfDecoding", {0});
        set("CameraFilmFormatMm", {8.0});
        set("CameraDiffusionEnabled", {index >= 2 ? 1 : 0});
        set("PrintDiffusionEnabled", {index == 1 || index == 3 ? 1 : 0});
        set("HalationActive", {index == 3 ? 1 : 0});
        set("HalationHighlightProtectionEv", {0.0});
        set("HalationHighlightBoostRange", {0.9});
        ofx_test::require_render(fixture.render() == kOfxStatOK, "highlight baseline render");
        const auto baseline = images.pixels(images.read());

        change_boost(2.0);
        images.reset();
        fixture.host().clear_calls();
        if (const auto status = fixture.render(); status != kOfxStatOK) {
            for (const auto& call : fixture.host().calls()) {
                if (call.operation == "message" && call.values.size() == 2) {
                    std::fprintf(stderr, "%s\n", std::get<std::string>(call.values[1]).c_str());
                }
            }
            throw std::runtime_error("active highlight reconstruction render status " + std::to_string(status));
        }
        const auto active = images.pixels(images.read());
        double maximumChange = 0.0;
        for (std::size_t i = 0; i < active.size(); ++i) {
            ofx_test::require_render(std::isfinite(active[i]), "finite highlight output");
            maximumChange = std::max(maximumChange, std::abs(static_cast<double>(active[i]) - baseline[i]));
        }
        ofx_test::require_render(maximumChange > 1e-4, "reconstructed carrier changes downstream output");
        images.reset();
        ofx_test::require_render(fixture.render() == kOfxStatOK && words(images.pixels(images.read())) == words(active),
                                 "same-frame highlight output is repeatable");

        if (index == 0) {
            images.reset();
            fixture.host().set_image_property(fixture.effect(), "Source", kOfxImagePropRegionOfDefinition, {x - 2, y, x + width, y + height});
            ofx_test::require_render(fixture.render() == kOfxStatErrFatal, "highlight boost rejects partial source RoD");
            for (float sample : images.read()) {
                RenderAssertions::require_same_bits("rejected highlight destination", sample, ofx_test::DeviceImages::kCanary);
            }
            fixture.host().set_image_property(fixture.effect(), "Source", kOfxImagePropRegionOfDefinition, {x, y, x + width, y + height});
        }
        change_boost(0.0);
        images.reset();
        ofx_test::require_render(fixture.render() == kOfxStatOK && words(images.pixels(images.read())) == words(baseline),
                                 "zero highlight boost returns identical route pixels");
    }

    constexpr std::array kGeometryRows{"format/rgb", "format/rgba", "geometry/padding", "geometry/pitch", "geometry/bounds", "geometry/coverage", "geometry/overlap"};
    void geometry_case(std::string_view name, const JuicerTests::Cohort& rows, const ofx_test::RenderPixels& expected) {
        const int components = name == "format/rgba" ? 4 : 3;
        const int x = name == "geometry/bounds" ? 11 : 0;
        const int y = name == "geometry/bounds" ? -3 : 0;
        const int destination_extra = name == "geometry/pitch" ? 13 : 5;
        const int border = name == "geometry/bounds" ? 1 : 0;
        ofx_test::DeviceImages images({7, 5, components, 5, destination_extra, border});
        ofx_test::RenderHost fixture(candidate());
        fixture.configure(images, {x, y});
        fixture.host().set_value(fixture.effect(), "ScanRoute", {std::string("NegativeDirectScan")});
        fixture.host().set_value(fixture.effect(), "GlareActive", {0});
        if (name == "geometry/coverage") {
            void* short_allocation = nullptr;
            constexpr std::size_t short_bytes = 4096;
            constexpr int advertised_row_bytes = 4096;
            ofx_test::require_cuda(cudaMalloc(&short_allocation, short_bytes), "short destination allocation");
            try {
                const std::vector<float> canaries(short_bytes / sizeof(float), ofx_test::DeviceImages::kCanary);
                ofx_test::require_cuda(cudaMemcpy(short_allocation, canaries.data(), short_bytes, cudaMemcpyHostToDevice), "initialize short destination");
                fixture.host().set_image_property(fixture.effect(), "Output", kOfxImagePropRowBytes, {advertised_row_bytes});
                fixture.host().set_image_property(fixture.effect(), "Output", kOfxImagePropData, {short_allocation});
                ofx_test::require_render(fixture.render() == kOfxStatErrFatal, "undersized allocation accepted");
                std::vector<float> contents(short_bytes / sizeof(float));
                ofx_test::require_cuda(cudaMemcpy(contents.data(), short_allocation, short_bytes, cudaMemcpyDeviceToHost), "read short destination");
                for (float value : contents)
                    RenderAssertions::require_same_bits("uncovered allocation canary", value, ofx_test::DeviceImages::kCanary);
            } catch (...) {
                (void)cudaFree(short_allocation);
                throw;
            }
            ofx_test::require_cuda(cudaFree(short_allocation), "release short destination");
        } else if (name == "geometry/overlap") {
            void* shared = nullptr;
            ofx_test::require_cuda(cudaMalloc(&shared, images.bytes() + images.destination_bytes()), "shared allocation");
            auto* bytes = static_cast<unsigned char*>(shared);
            try {
                ofx_test::require_cuda(cudaMemcpy(shared, images.source(), images.bytes(), cudaMemcpyDeviceToDevice), "copy shared source");
                ofx_test::require_cuda(cudaMemcpy(bytes + images.bytes(), images.destination(), images.destination_bytes(), cudaMemcpyDeviceToDevice), "initialize shared destination");
                fixture.host().set_image_property(fixture.effect(), "Source", kOfxImagePropData, {shared});
                fixture.host().set_image_property(fixture.effect(), "Output", kOfxImagePropData, {static_cast<void*>(bytes + images.bytes())});
                auto disjoint = rows.case_event("overlap/disjoint");
                ofx_test::require_render(fixture.render() == kOfxStatOK, "disjoint ranges in shared allocation rejected");
                ofx_test::require_cuda(cudaDeviceSynchronize(), "complete disjoint render");
                ofx_test::require_cuda(cudaMemcpy(images.destination(), bytes + images.bytes(), images.destination_bytes(), cudaMemcpyDeviceToDevice), "read shared destination");
                RenderAssertions::compare_pixels("disjoint shared allocation", images.pixels(images.read()), expected.at("route/negative-direct"));
                disjoint.passed();
                auto partial = rows.case_event("overlap/partial");
                std::vector<float> before(images.bytes() / sizeof(float));
                ofx_test::require_cuda(cudaMemcpy(before.data(), shared, images.bytes(), cudaMemcpyDeviceToHost), "snapshot overlapping source");
                fixture.host().set_image_property(fixture.effect(), "Output", kOfxImagePropData, {static_cast<void*>(bytes + sizeof(float))});
                ofx_test::require_render(fixture.render() == kOfxStatErrFatal, "partially overlapping ranges accepted");
                std::vector<float> after(before.size());
                ofx_test::require_cuda(cudaMemcpy(after.data(), shared, images.bytes(), cudaMemcpyDeviceToHost), "read rejected overlap");
                RenderAssertions::compare_bits("overlap source unchanged", after, before);
                partial.passed();
            } catch (...) {
                (void)cudaFree(shared);
                throw;
            }
            ofx_test::require_cuda(cudaFree(shared), "release shared allocation");
        } else {
            const auto status = fixture.render();
            if (status != kOfxStatOK)
                throw std::runtime_error(std::string(name) + ": geometry render status " + std::to_string(status));
            const auto actual = images.pixels(images.read());
            for (float value : actual)
                ofx_test::require_render(std::isfinite(value) && value >= 0.0f && value <= 1.0f, "geometry pixel outside range");
            const auto& predicted = expected.at(name == "format/rgba" || name == "geometry/bounds" ? std::string(name) : "route/negative-direct");
            RenderAssertions::compare_pixels(std::string(name), actual, predicted);
            if (components == 4)
                for (std::size_t alpha = 3; alpha < actual.size(); alpha += 4)
                    RenderAssertions::require_same_bits("geometry alpha", actual[alpha], predicted[alpha]);
        }
        if (name == "format/rgba") {
            images.reset();
            fixture.host().set_image_property(fixture.effect(), "Output", kOfxImageEffectPropComponents, {std::string("OfxImageComponentUnknown")});
            fixture.host().set_property(fixture.host().action_args(), kOfxImageEffectPropRenderWindow, {0, 0, 0, 5});
            ofx_test::require_render(fixture.render() == kOfxStatOK, "empty window rejected unknown destination format");
            fixture.host().set_property(fixture.host().action_args(), kOfxImageEffectPropRenderWindow, {0, 0, 7, 5});
            fixture.host().clear_calls();
            ofx_test::require_render(fixture.render() == kOfxStatErrFatal, "nonempty unknown destination format accepted");
            const auto& calls = fixture.host().calls();
            ofx_test::require_render(std::any_of(calls.begin(), calls.end(), [](const auto& call) {
                                         return call.operation == "get_double" && call.property == kOfxImageEffectPropFrameRate;
                                     }),
                                     "unknown destination rejected before frame assembly");
            for (float sample : images.read())
                RenderAssertions::require_same_bits("unknown destination unchanged", sample, ofx_test::DeviceImages::kCanary);
        }
        if (name == "format/rgb") {
            images.reset();
            fixture.host().set_image_property(fixture.effect(), "Source", kOfxImageEffectPropComponents, {std::string(kOfxImageComponentAlpha)});
            fixture.host().set_property(fixture.host().action_args(), kOfxImageEffectPropRenderWindow, {0, 0, 0, 5});
            ofx_test::require_render(fixture.render() == kOfxStatOK, "Alpha source preserves empty-window success");
            fixture.host().set_property(fixture.host().action_args(), kOfxImageEffectPropRenderWindow, {0, 0, 7, 5});
            fixture.host().clear_calls();
            ofx_test::require_render(fixture.render() == kOfxStatErrFatal, "Alpha source rejected at native format boundary");
            ofx_test::require_render(std::any_of(fixture.host().calls().begin(), fixture.host().calls().end(), [](const auto& call) {
                                         return call.operation == "get_double" && call.property == kOfxImageEffectPropFrameRate;
                                     }),
                                     "Alpha source reaches frame assembly");
            for (float value : images.read()) {
                RenderAssertions::require_same_bits("Alpha rejection destination", value, ofx_test::DeviceImages::kCanary);
            }
        }
        const std::size_t releases = (name == "format/rgb" || name == "format/rgba") ? 3 : name == "geometry/overlap" ? 2
                                                                                                                      : 1;
        ofx_test::require_render(fixture.host().image_release_count(fixture.effect(), "Source") == releases &&
                                     fixture.host().image_release_count(fixture.effect(), "Output") == releases,
                                 "geometry image release count");
    }
    constexpr std::array kStreamRows{"stream/missing", "stream/null", "stream/explicit"};
    void stream_case(std::string_view name, const ofx_test::RenderPixels& expected) {
        ofx_test::DeviceImages images({7, 5, 3});
        ofx_test::RenderHost fixture(candidate());
        fixture.configure(images);
        fixture.host().set_value(fixture.effect(), "ScanRoute", {std::string("NegativeDirectScan")});
        if (name == "stream/missing")
            fixture.host().set_property(fixture.host().action_args(), kOfxImageEffectPropCudaStream, {});
        else if (name == "stream/null")
            fixture.host().set_property(fixture.host().action_args(), kOfxImageEffectPropCudaStream, {static_cast<void*>(nullptr)});
        ofx_test::require_render(fixture.render() == kOfxStatOK, "stream render failed");
        RenderAssertions::compare_pixels(std::string(name), images.pixels(images.read()), expected.at("route/negative-direct"));
        ofx_test::require_render(fixture.host().image_release_count(fixture.effect(), "Source") == 1 &&
                                     fixture.host().image_release_count(fixture.effect(), "Output") == 1,
                                 "stream image release count");
    }
    void wrong_context_case() {
        ofx_test::DeviceImages images({7, 5, 3});
        ofx_test::RenderHost fixture(candidate());
        fixture.configure(images);
        CUcontext current = nullptr;
        CUcontext other = nullptr;
        CUstream foreign = nullptr;
        ofx_test::require_render(cuCtxGetCurrent(&current) == CUDA_SUCCESS && current, "current CUDA context");
        ofx_test::require_render(cuCtxCreate(&other, nullptr, 0, 0) == CUDA_SUCCESS, "create other CUDA context");
        try {
            ofx_test::require_render(cuStreamCreate(&foreign, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS, "create foreign stream");
            ofx_test::require_render(cuCtxSetCurrent(current) == CUDA_SUCCESS, "restore CUDA context");
            fixture.host().set_property(fixture.host().action_args(), kOfxImageEffectPropCudaStream, {reinterpret_cast<void*>(foreign)});
            ofx_test::require_render(fixture.render() == kOfxStatErrFatal, "foreign context stream accepted");
            for (float value : images.read())
                RenderAssertions::require_same_bits("foreign stream destination", value, ofx_test::DeviceImages::kCanary);
            ofx_test::require_render(fixture.host().image_release_count(fixture.effect(), "Source") == 1 &&
                                         fixture.host().image_release_count(fixture.effect(), "Output") == 1,
                                     "foreign stream image releases");
            ofx_test::require_render(cuCtxSetCurrent(other) == CUDA_SUCCESS, "select foreign context");
            ofx_test::require_render(cuStreamDestroy(foreign) == CUDA_SUCCESS, "destroy foreign stream");
            foreign = nullptr;
            ofx_test::require_render(cuCtxDestroy(other) == CUDA_SUCCESS, "destroy foreign context");
            other = nullptr;
            ofx_test::require_render(cuCtxSetCurrent(current) == CUDA_SUCCESS, "restore original context");
        } catch (...) {
            if (other) {
                (void)cuCtxSetCurrent(other);
                if (foreign)
                    (void)cuStreamDestroy(foreign);
                (void)cuCtxDestroy(other);
            }
            (void)cuCtxSetCurrent(current);
            throw;
        }
    }
    void owners_case(bool teardown) {
        ofx_test::DeviceImages first({7, 5, 3});
        ofx_test::DeviceImages second({7, 5, 3});
        ofx_test::RenderHost fixture(candidate());
        fixture.configure(first);
        auto* other = fixture.host().create_effect();
        ofx_test::require_render(fixture.call(kOfxActionCreateInstance, other) == kOfxStatOK, "create second instance");
        ofx_test::require_render(fixture.render() == kOfxStatOK, "first instance render");
        const auto first_pixels = first.pixels(first.read());
        fixture.configure_effect(other, second);
        ofx_test::require_render(fixture.call(kOfxImageEffectActionRender, other, fixture.host().action_args()) == kOfxStatOK, "second instance render");
        RenderAssertions::compare_pixels("two instances", second.pixels(second.read()), first_pixels);
        ofx_test::require_render(fixture.call(kOfxActionDestroyInstance, other) == kOfxStatOK, "destroy second instance");
        ofx_test::require_render(fixture.host().image_release_count(other, "Source") == 1 &&
                                     fixture.host().image_release_count(other, "Output") == 1,
                                 "second instance releases");
        if (teardown) {
            fixture.configure(first);
            first.reset();
            ofx_test::require_render(fixture.render() == kOfxStatOK, "survivor after peer teardown");
            RenderAssertions::compare_pixels("survivor pixels", first.pixels(first.read()), first_pixels);
            fixture.close();
        }
    }
    void cancellation_case(const ofx_test::RenderPixels& expected) {
        ofx_test::DeviceImages images({7, 5, 3});
        ofx_test::RenderHost fixture(candidate());
        fixture.configure(images);
        fixture.host().set_value(fixture.effect(), "ScanRoute", {std::string("NegativeDirectScan")});
        fixture.host().set_abort(true);
        ofx_test::require_render(fixture.render() == kOfxStatOK, "cancellation status");
        const auto stopped = images.read();
        for (float value : stopped)
            RenderAssertions::require_same_bits("cancelled destination", value, ofx_test::DeviceImages::kCanary);
        fixture.host().set_abort(false);
        ofx_test::require_render(fixture.render() == kOfxStatOK, "render after cancellation");
        RenderAssertions::compare_pixels("render after cancellation", images.pixels(images.read()), expected.at("route/negative-direct"));
        ofx_test::require_render(fixture.host().image_release_count(fixture.effect(), "Source") == 2 &&
                                     fixture.host().image_release_count(fixture.effect(), "Output") == 2,
                                 "cancellation image releases");
    }
} // namespace

int main(int argc, char** argv) try {
    if (JuicerTests::list_requested(argc, argv)) {
        for (const auto& route : kRoutes)
            JuicerTests::list_case(kCaller, route.id);
        for (const auto* id : kBoostRows)
            JuicerTests::list_case(kCaller, id);
        for (const auto* id : kGeometryRows)
            JuicerTests::list_case(kCaller, id);
        for (const auto* id : kStreamRows)
            JuicerTests::list_case(kCaller, id);
        JuicerTests::list_case(kCaller, "overlap/disjoint");
        JuicerTests::list_case(kCaller, "overlap/partial");
        JuicerTests::list_case(kCaller, "stream/wrong-context");
        JuicerTests::list_case(kCaller, "cancel/then_success");
        JuicerTests::list_case(kCaller, "owners/two_instances");
        JuicerTests::list_case(kCaller, "owners/teardown");
        return 0;
    }
    ofx_test::require_render(argc == 3 && std::string_view(argv[1]) == "--reference", "usage: GPU render fixture --reference file");
    JuicerTests::CaseLog log;
    const auto expected = ofx_test::read_render_reference(argv[2], JUICER_SPECTRAL_TRACE_PRESET, true);
    std::vector<std::string> ids;
    ids.reserve(20);
    for (const auto& route : kRoutes)
        ids.emplace_back(route.id);
    for (const auto* id : kBoostRows)
        ids.emplace_back(id);
    for (const auto* id : kGeometryRows)
        ids.emplace_back(id);
    for (const auto* id : kStreamRows)
        ids.emplace_back(id);
    ids.emplace_back("overlap/disjoint");
    ids.emplace_back("overlap/partial");
    ids.emplace_back("stream/wrong-context");
    ids.emplace_back("cancel/then_success");
    ids.emplace_back("owners/two_instances");
    ids.emplace_back("owners/teardown");
    JuicerTests::Cohort rows(std::string(kCaller), ids);
    for (const auto& route : kRoutes) {
        auto row = rows.case_event(route.id);
        route_case(route, expected);
        row.passed();
    }
    for (std::size_t i = 0; i < kBoostRows.size(); ++i) {
        auto row = rows.case_event(kBoostRows[i]);
        highlight_case(i);
        row.passed();
    }
    for (const auto* id : kGeometryRows) {
        auto row = rows.case_event(id);
        geometry_case(id, rows, expected);
        row.passed();
    }
    for (const auto* id : kStreamRows) {
        auto row = rows.case_event(id);
        stream_case(id, expected);
        row.passed();
    }
    {
        auto row = rows.case_event("stream/wrong-context");
        wrong_context_case();
        row.passed();
    }
    {
        auto row = rows.case_event("cancel/then_success");
        cancellation_case(expected);
        row.passed();
    }
    for (const auto* id : {"owners/two_instances", "owners/teardown"}) {
        auto row = rows.case_event(id);
        owners_case(std::string_view(id) == "owners/teardown");
        row.passed();
    }
    log.finish();
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "GPU render fixture: %s\n", error.what());
    return 1;
}
