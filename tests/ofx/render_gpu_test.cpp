#include "render_gpu_fixture.h"
#include "render_assertions.h"
#include "../ffi/case_results.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cuda.h>
#include <cstdio>
#include <exception>
#include <map>
#include "ofxsImageEffect.h"
#include <string>
#include <vector>

namespace {
    constexpr std::string_view kCaller = "ofx.gpu.render";
    struct Route {
        const char* id;
        const char* route;
        const char* profile;
        int reconstruction;
        int components;
        int x;
        int y;
    };
    constexpr std::array kRoutes{
        Route{"route/negative-direct", "NegativeDirectScan", "kodak_portra_400", 0, 3, 0, 0},
        Route{"route/negative-print", "NegativePrintScan", "kodak_portra_400", 1, 4, 0, 0},
        Route{"route/positive-direct", "NegativeDirectScan", "fujifilm_provia_100f", 2, 3, 11, -3},
        Route{"route/positive-print", "NegativePrintScan", "fujifilm_provia_100f", 0, 4, 0, 0}};

    std::map<std::string, std::vector<float>> reference_prints() {
        ofx_test::RenderHost reference(OfxGetPlugin(0));
        std::map<std::string, std::vector<float>> result;
        for (std::size_t index : {std::size_t{1}, std::size_t{3}}) {
            const auto& route = kRoutes[index];
            ofx_test::DeviceImages images({7, 5, route.components});
            auto* effect = index == 1 ? reference.effect() : reference.host().create_effect();
            if (index != 1)
                ofx_test::require_render(reference.call(kOfxActionCreateInstance, effect) == kOfxStatOK, "reference second instance");
            reference.configure_effect(effect, images, {route.x, route.y});
            reference.host().set_value(effect, "ScanRoute", {std::string(route.route)});
            reference.host().set_value(effect, "FilmProfileKey", {std::string(route.profile)});
            reference.host().set_value(effect, "SpectralUpsampling", {route.reconstruction});
            ofx_test::require_render(reference.call(kOfxImageEffectActionRender, effect, reference.host().action_args()) == kOfxStatOK,
                                     "independent reference print render");
            result.emplace(route.id, images.pixels(images.read()));
            ofx_test::require_render(reference.host().image_release_count(effect, "Source") == 1 &&
                                         reference.host().image_release_count(effect, "Output") == 1,
                                     "reference image release count");
            if (index != 1)
                ofx_test::require_render(reference.call(kOfxActionDestroyInstance, effect) == kOfxStatOK, "reference second teardown");
        }
        reference.close();
        return result;
    }
    void route_case(const Route& route, const std::map<std::string, std::vector<float>>& prints) {
        ofx_test::DeviceImages images({7, 5, route.components});
        ofx_test::RenderHost fixture;
        fixture.configure(images, {route.x, route.y});
        fixture.host().set_value(fixture.effect(), "ScanRoute", {std::string(route.route)});
        fixture.host().set_value(fixture.effect(), "FilmProfileKey", {std::string(route.profile)});
        fixture.host().set_value(fixture.effect(), "SpectralUpsampling", {route.reconstruction});
        fixture.host().set_value(fixture.effect(), "GlareActive", {0});
        const bool print = std::string_view(route.route) == "NegativePrintScan";
        if (print) {
            fixture.host().set_property(fixture.host().action_args(), kOfxImageEffectPropRenderWindow, {route.x, route.y, route.x + 6, route.y + 5});
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
            fixture.host().set_property(fixture.host().action_args(), kOfxImageEffectPropRenderWindow, {route.x, route.y, route.x + 7, route.y + 5});
        }
        ofx_test::require_render(fixture.render() == kOfxStatOK, "route render failed");
        const auto actual = images.pixels(images.read());
        for (float value : actual)
            ofx_test::require_render(std::isfinite(value) && value >= 0.0f && value <= 1.0f, "route pixel outside finite output range");
        if (std::string_view(route.route) == "NegativeDirectScan")
            RenderAssertions::compare_pixels(route.id, actual, ofx_test::reference_pixels(JUICER_PROCESSOR_RUST_PATH, std::string_view(route.id).substr(6)));
        else
            RenderAssertions::compare_pixels(route.id, actual, prints.at(route.id));
        const std::size_t releases = print ? 2 : 1;
        ofx_test::require_render(fixture.host().image_release_count(fixture.effect(), "Source") == releases &&
                                     fixture.host().image_release_count(fixture.effect(), "Output") == releases,
                                 "route image release count");
    }
    constexpr std::array kGeometryRows{"format/rgb", "format/rgba", "geometry/padding", "geometry/pitch", "geometry/bounds", "geometry/coverage", "geometry/overlap"};
    void geometry_case(std::string_view name, const JuicerTests::Cohort& rows) {
        const int components = name == "format/rgba" ? 4 : 3;
        const int x = name == "geometry/bounds" ? 11 : 0;
        const int y = name == "geometry/bounds" ? -3 : 0;
        const int destination_extra = name == "geometry/pitch" ? 13 : 5;
        const int border = name == "geometry/bounds" ? 1 : 0;
        ofx_test::DeviceImages images({7, 5, components, 5, destination_extra, border});
        ofx_test::RenderHost fixture;
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
                RenderAssertions::compare_pixels("disjoint shared allocation", images.pixels(images.read()), ofx_test::reference_pixels(JUICER_PROCESSOR_RUST_PATH, "negative-direct"));
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
            if (components == 3) {
                RenderAssertions::compare_pixels(std::string(name), actual, ofx_test::reference_pixels(JUICER_PROCESSOR_RUST_PATH, "negative-direct"));
            } else {
                std::vector<float> rgb;
                rgb.reserve(actual.size() / 4 * 3);
                for (std::size_t i = 0; i < actual.size(); i += 4)
                    rgb.insert(rgb.end(), actual.begin() + static_cast<std::ptrdiff_t>(i), actual.begin() + static_cast<std::ptrdiff_t>(i + 3));
                RenderAssertions::compare_pixels(std::string(name), rgb, ofx_test::reference_pixels(JUICER_PROCESSOR_RUST_PATH, "negative-direct"));
            }
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
    void stream_case(std::string_view name) {
        ofx_test::DeviceImages images({7, 5, 3});
        ofx_test::RenderHost fixture;
        fixture.configure(images);
        fixture.host().set_value(fixture.effect(), "ScanRoute", {std::string("NegativeDirectScan")});
        if (name == "stream/missing")
            fixture.host().set_property(fixture.host().action_args(), kOfxImageEffectPropCudaStream, {});
        else if (name == "stream/null")
            fixture.host().set_property(fixture.host().action_args(), kOfxImageEffectPropCudaStream, {static_cast<void*>(nullptr)});
        ofx_test::require_render(fixture.render() == kOfxStatOK, "stream render failed");
        RenderAssertions::compare_pixels(std::string(name), images.pixels(images.read()), ofx_test::reference_pixels(JUICER_PROCESSOR_RUST_PATH, "negative-direct"));
        ofx_test::require_render(fixture.host().image_release_count(fixture.effect(), "Source") == 1 &&
                                     fixture.host().image_release_count(fixture.effect(), "Output") == 1,
                                 "stream image release count");
    }
    void wrong_context_case() {
        ofx_test::DeviceImages images({7, 5, 3});
        ofx_test::RenderHost fixture;
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
        ofx_test::RenderHost fixture;
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
    void cancellation_case() {
        ofx_test::DeviceImages images({7, 5, 3});
        ofx_test::RenderHost fixture;
        fixture.configure(images);
        fixture.host().set_value(fixture.effect(), "ScanRoute", {std::string("NegativeDirectScan")});
        fixture.host().set_abort(true);
        ofx_test::require_render(fixture.render() == kOfxStatOK, "cancellation status");
        const auto stopped = images.read();
        for (float value : stopped)
            RenderAssertions::require_same_bits("cancelled destination", value, ofx_test::DeviceImages::kCanary);
        fixture.host().set_abort(false);
        ofx_test::require_render(fixture.render() == kOfxStatOK, "render after cancellation");
        RenderAssertions::compare_pixels("render after cancellation", images.pixels(images.read()), ofx_test::reference_pixels(JUICER_PROCESSOR_RUST_PATH, "negative-direct"));
        ofx_test::require_render(fixture.host().image_release_count(fixture.effect(), "Source") == 2 &&
                                     fixture.host().image_release_count(fixture.effect(), "Output") == 2,
                                 "cancellation image releases");
    }
} // namespace

int main(int argc, char** argv) try {
    if (JuicerTests::list_requested(argc, argv)) {
        for (const auto& route : kRoutes)
            JuicerTests::list_case(kCaller, route.id);
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
    JuicerTests::CaseLog log;
    const auto prints = reference_prints();
    std::vector<std::string> ids;
    ids.reserve(20);
    for (const auto& route : kRoutes)
        ids.emplace_back(route.id);
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
        route_case(route, prints);
        row.passed();
    }
    for (const auto* id : kGeometryRows) {
        auto row = rows.case_event(id);
        geometry_case(id, rows);
        row.passed();
    }
    for (const auto* id : kStreamRows) {
        auto row = rows.case_event(id);
        stream_case(id);
        row.passed();
    }
    {
        auto row = rows.case_event("stream/wrong-context");
        wrong_context_case();
        row.passed();
    }
    {
        auto row = rows.case_event("cancel/then_success");
        cancellation_case();
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
