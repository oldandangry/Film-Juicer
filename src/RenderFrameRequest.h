#pragma once

#include <cstdint>
#include <memory>

#include "OutputColor.h"
#include "Print.h"
#include "ProfileJSONLoader.h"
#include "RenderRecipe.h"
#include "Scanner.h"
#include "ofxsProcessing.h"

struct WorkingState;
struct DirectRenderState;
struct PrintRenderState;

// Broad frame request data is retained only for non-focused host plumbing; product
// rendering consumes typed direct/print requests.
struct FrameRequest {
    std::shared_ptr<const RenderRecipe> recipe;
    std::shared_ptr<const ::WorkingState> workingState;
    const Print::Runtime* printRt = nullptr;
    bool workingStateReady = false;
    bool printRtReady = false;

    int components = 0;
    OfxRectI renderWindow{0, 0, 0, 0};
    Scanner::Options scannerOptions;
    Scanner::Settings scannerSettings;
    Print::Params printParams;
    Profiles::HalationMetadata halationOverride{};
    bool hasHalationOverride = false;
    Profiles::ProfileGlare printGlareOverride{};
    bool hasPrintGlareOverride = false;
    float exposureScale = 1.0f;
    bool cameraAutoEnabled = false;
    int cameraMeteringMethod = 0;
    double cameraSliderEV = 0.0;
    OfxRectI autoExposureMeterBounds{0, 0, 0, 0};
    bool autoExposureMeterBoundsValid = false;
    OutputEncoding::Params outputEncoding;
    std::uint64_t sessionSeed = 1;
    std::uint64_t instanceToken = 1;
    std::uintptr_t clipToken = 0;
    double frameTime = 0.0;
    double frameRate = 0.0;
    std::uint32_t frameBoundsVersion = 0;
    float pixelSizeUm = 0.0f;
    bool interactiveRenderStatus = false;
    bool renderQualityDraft = false;
    bool sequentialRenderStatus = false;
};

struct DirectFrameRequest {
    std::shared_ptr<const DirectRenderState> state;
    int components = 0;
    OfxRectI renderWindow{0, 0, 0, 0};
    OfxRectI fullFrameExtent{0, 0, 0, 0};
    std::uint64_t sessionSeed = 1;
    std::uint64_t instanceToken = 1;
    std::uintptr_t clipToken = 0;
    double frameTime = 0.0;
    double frameRate = 0.0;
    float pixelSizeUm = 0.0f;
};

struct PrintFrameRequest {
    std::shared_ptr<const PrintRenderState> state;
    int components = 0;
    OfxRectI renderWindow{0, 0, 0, 0};
    OfxRectI fullFrameExtent{0, 0, 0, 0};
    std::uint64_t sessionSeed = 1;
    std::uint64_t instanceToken = 1;
    std::uintptr_t clipToken = 0;
    double frameTime = 0.0;
    double frameRate = 0.0;
    float pixelSizeUm = 0.0f;
};

namespace Spektrafilm {

    using ::DirectFrameRequest;
    using ::FrameRequest;
    using ::PrintFrameRequest;

} // namespace Spektrafilm
