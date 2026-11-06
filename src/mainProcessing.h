// JuicerProcessing.h
#pragma once

#include <cstdint>

#include "ofxsImageEffect.h"
#include "ofxsProcessing.h"

#include "RenderValues.h"
#include "juicer_cuda_frame_values.h"
#include "juicer_legacy_api.h"

struct NativeSubmissionState;
class PreparedState;

// Full class declaration
class JuicerProcessor : public OFX::ImageProcessor {
public:
    JuicerProcessor(OFX::ImageEffect& effect, FjRuntimeView runtime);

    struct SourceDestinationImages {
        OFX::Image* src = nullptr;
        OFX::Image* dst = nullptr;
    };

    struct DirectFrameRequest {
        // Borrowed through the synchronous process() call; never retained beyond it.
        const PreparedState* renderState = nullptr;
        FjPreparedFramePolicy framePolicy{};
        OfxRectI renderWindow{0, 0, 0, 0};
        Spektrafilm::FilmJuicerEffectsGeometry effectsGeometry{};
        OfxRectI fullFrameExtent{0, 0, 0, 0};
        std::uint64_t sessionSeed = 1;
        std::uint64_t instanceToken = 1;
        std::uintptr_t clipToken = 0;
        double frameTime = 0.0;
        double frameRate = 0.0;
    };

    struct PrintFrameRequest {
        // Borrowed through the synchronous process() call; never retained beyond it.
        const PreparedState* renderState = nullptr;
        FjPreparedFramePolicy framePolicy{};
        OfxRectI renderWindow{0, 0, 0, 0};
        Spektrafilm::FilmJuicerEffectsGeometry effectsGeometry{};
        OfxRectI fullFrameExtent{0, 0, 0, 0};
        std::uint64_t sessionSeed = 1;
        std::uint64_t instanceToken = 1;
        std::uintptr_t clipToken = 0;
        double frameTime = 0.0;
        double frameRate = 0.0;
    };

    void setSrcDst(const SourceDestinationImages& images);
    void setDirectFrameRequest(const DirectFrameRequest& request);
    void setPrintFrameRequest(const PrintFrameRequest& request);
    void setSubmissionState(NativeSubmissionState* s);

    void process() override;
    void processImagesCUDA() override;

private:
    // SF_TEMP_BRIDGE_RUNTIME_CALL_VIEW: same callback Runtime; remove S6.
    FjRuntimeView _runtime;
    OFX::Image* _srcImg = nullptr;
    const PreparedState* _renderState = nullptr;
    NativeSubmissionState* _submissionState = nullptr;
    std::uint64_t _sessionSeed = 1;
    std::uint64_t _instanceToken = 1;
    std::uintptr_t _clipToken = 0;
    std::int64_t _frameIndex = 0;
    double _timeFrames = 0.0;
    double _frameRate = 0.0;
    Spektrafilm::FilmJuicerEffectsGeometry _effectsGeometry{};
    OfxRectI _fullFrameExtent{0, 0, 0, 0};
    FjPreparedFramePolicy _framePolicy{};
};
