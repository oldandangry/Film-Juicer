// JuicerProcessing.h
#pragma once

#include <cstdint>
#include <memory>

#include "ofxsImageEffect.h"
#include "ofxsProcessing.h"

#include "RenderRecipe.h"
#include "juicer_cuda_frame_values.h"

struct InstanceState;
struct DirectRenderState;
struct PrintRenderState;

// Full class declaration
class JuicerProcessor : public OFX::ImageProcessor {
public:
    explicit JuicerProcessor(OFX::ImageEffect& effect);

    struct SourceDestinationImages {
        OFX::Image* src = nullptr;
        OFX::Image* dst = nullptr;
    };

    struct DirectFrameRequest {
        std::shared_ptr<const DirectRenderState> state;
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
        std::shared_ptr<const PrintRenderState> state;
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
    void setInstanceState(InstanceState* s);

    void process() override;
    void processImagesCUDA() override;

private:
    OFX::Image* _srcImg = nullptr;
    std::shared_ptr<const DirectRenderState> _directStateHold;
    std::shared_ptr<const PrintRenderState> _printStateHold;
    InstanceState* _instanceState = nullptr;
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
