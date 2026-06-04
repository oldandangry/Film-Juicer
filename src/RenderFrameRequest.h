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

// Phase 1C request bridge ledger:
// - SF_TEMP_BRIDGE_FrameRequestSideChannelCopy: JuicerProcessor::setFrameRequest still copies this
//   immutable boundary into mutable processor members while Phase 1A blocks product pixels.
//   Allowed call-site family: JuicerEffect::render adapter to JuicerProcessor. Disposition:
//   .tmp/spektrafilm-phase-1C-frame-request-disposition.md. Removal starts in Phase 3 direct route
//   and continues through the owning print/DIR/optics/scanner/grain phases.
// - SF_TEMP_BRIDGE_CPUAutoExposureBlocked: current CPU auto-exposure fields are retained only as
//   blocked legacy request facts; product spektrafilm rendering is cut off before CPU pixel reads.
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

namespace Spektrafilm {

    using ::FrameRequest;

} // namespace Spektrafilm
