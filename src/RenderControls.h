#pragma once

#include <array>

struct DirCouplersControls {
    bool active = true;
    float amount = 1.0f;
    float inhibitionSameLayer = 1.0f;
    float inhibitionInterlayer = 1.0f;
    float diffusionSizeUm = 20.0f;
    float diffusionTailUm = 200.0f;
    float diffusionTailWeight = 0.03f;
    bool gammaUseStock = true;
    std::array<float, 3> gammaSameLayerRgb{{0.341f, 0.324f, 0.273f}};
    std::array<float, 2> gammaInterlayerRToGb{{0.355f, 0.305f}};
    std::array<float, 2> gammaInterlayerGToRb{{0.154f, 0.358f}};
    std::array<float, 2> gammaInterlayerBToRg{{0.171f, 0.225f}};
    std::array<float, 3> langmuirDonorKRgb{{1.0f, 1.0f, 1.0f}};
    std::array<float, 3> langmuirReceiverKRgb{{1.0f, 1.0f, 1.0f}};
};

namespace Spektrafilm {
    using ::DirCouplersControls;
}

// Captured resolved controls; mapping/default conversion transfers to S6.B/C, boundary removed S6.D.
struct VisualGrainControls {
    bool active = false;
    bool sublayersActive = true;
    float particleAreaUm2 = 0.2f;
    std::array<float, 3> particleScaleCmy{{0.8f, 1.0f, 2.0f}};
    std::array<float, 3> particleScaleLayers{{2.5f, 1.0f, 0.5f}};
    std::array<float, 3> visualParticleDensityMinCmy{{0.07f, 0.08f, 0.12f}};
    std::array<float, 3> uniformityCmy{{0.97f, 0.97f, 0.99f}};
    float correlationSigmaPx = 0.65f;
    float dyeCloudBlurUm = 1.0f;
    std::array<float, 2> microStructure{{0.2f, 30.0f}};
    int nSubLayers = 1;
    float amplitude = 1.0f;
    float chromaMix = 0.0f;
    float chromaSharedWeight = 1.0f;
    float chromaIndependentWeight = 0.0f;
    float coarseWeight = 0.0f;
    float midWeight = 0.0f;
    float sizeMixScale = 1.0f;
    float clumpTemporalMix = 0.0f;
    float clumpMorphPeriodSec = 8.0f;
    int debugView = 0;
};

namespace Spektrafilm {
    using ::VisualGrainControls;
}
