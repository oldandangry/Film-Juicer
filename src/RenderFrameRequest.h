#pragma once

#include <cstdint>
#include <memory>

#include "Couplers.h"
#include "OutputColor.h"
#include "Print.h"
#include "ProfileJSONLoader.h"
#include "RenderRecipe.h"
#include "Scanner.h"
#include "ofxsProcessing.h"

struct WorkingState;
struct DirectRenderState;
struct PrintRenderState;

// Phase 1C request bridge ledger:
// - SF_TEMP_BRIDGE_FrameRequestSideChannelCopy owner=Phase4-print-route:
//   reason=retained Phase 4-only broad adapter; allowed=FrameRequest declaration and
//   JuicerProcessor::setFrameRequest only; output_impact=blocked print route; hash_impact=none;
//   resource_impact=broad legacy preparation; removal=Phase4 print cutover.
// - SF_TEMP_BRIDGE_CPUAutoExposureBlocked owner=Phase4-print-route:
//   reason=legacy non-direct request facts; allowed=JuicerProcessor::setFrameRequest only;
//   output_impact=blocked non-direct route; hash_impact=none; resource_impact=none;
//   removal=Phase4 print-route request cutover.
struct FrameRequest {
    std::shared_ptr<const RenderRecipe> recipe;
    std::shared_ptr<const ::WorkingState> workingState;
    const Print::Runtime* printRuntime = nullptr;
    bool workingStateReady = false;
    bool printRuntimeReady = false;

    int components = 0;
    OfxRectI renderWindow{0, 0, 0, 0};
    Scanner::Options scannerOptions;
    Scanner::Settings scannerSettings;
    Print::Params printParams;
    Profiles::HalationMetadata halationOverride{};
    bool hasHalationOverride = false;
    Profiles::GrainMetadata grainOverride{};
    bool hasGrainOverride = false;
    Profiles::ProfileGlare printGlareOverride{};
    bool hasPrintGlareOverride = false;
    // SF_TEMP_BRIDGE_CouplersLiveOfxRuntime owner=Phase4-print-route:
    // reason=legacy broad DIR request field; allowed=FrameRequest::dirRuntime declaration and
    // JuicerProcessor::setFrameRequest only;
    // output_impact=blocked print route;
    // hash_impact=none on direct route; resource_impact=legacy print DIR runtime; removal=Phase4.
    Couplers::Runtime dirRuntime;
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
    double gateWeaveAmount = 1.0;
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
