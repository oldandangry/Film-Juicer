#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "Cuda/JuicerCudaPayloads.h"
#include "FilmEffectsFrameDescriptors.h"
#include "RenderRecipe.h"

namespace JuicerCuda {

    struct FilmPreparedView {
        DeviceCurveView finalSensB{};
        DeviceCurveView finalSensG{};
        DeviceCurveView finalSensR{};
        DeviceCurveView normalizedDensB{};
        DeviceCurveView normalizedDensG{};
        DeviceCurveView normalizedDensR{};
        DeviceCurveView dirDensB{};
        DeviceCurveView dirDensG{};
        DeviceCurveView dirDensR{};
        const float* tablesIllum = nullptr;
        int tablesK = 0;
        const float* filmTcLut = nullptr;
        int filmTcLutExtent = 0;
        const float* mallettBasis = nullptr;
        int mallettBasisK = 0;
        float inputRGBToXYZ[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        float inputXYZAdapt[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        float xyzToLinearSrgb[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        int applyInputChromaticAdapt = 0;
        std::uint64_t finalSensitivityHash = 0;
        std::uint64_t normalizedDensityCurvesHash = 0;
        std::uint64_t dirCouplersHash = 0;
    };

    struct FilmPayloadPack {
        FilmRawPayload filmRaw{};
        FilmExposurePayload filmExposure{};
        FilmDevelopPayload filmDevelop{};
        std::uint64_t densityBoundsHash = 0;
    };

    // Invocation-local resolved coefficients. Density tables remain in the resource input.
    struct FilmPayloadInput {
        int inputColorSpace = 0;
        bool inputCctfDecoding = false;
        Spektrafilm::RgbToRawMethod method = Spektrafilm::RgbToRawMethod::Hanatos2025;
        float manualExposureEv = 0;
        float mallettGreenMidgrayScale = 1;
        std::uint64_t sensitivityHash = 0;
        std::uint64_t densityCurvesHash = 0;
        std::size_t densitySampleCount = 0;
        std::array<float, 3> gammaRgb{};
        DirNonlinearMode dirMode = DirNonlinearMode::Inactive;
        std::array<std::array<float, 3>, 3> dirMatrixRgb{};
        std::array<float, 3> densityMaxRgb{};
        std::array<float, 3> densityRefRgb{};
        std::array<float, 3> donorKRgb{};
        std::array<float, 3> receiverCRefRgb{};
        std::array<float, 3> receiverKrRgb{};
        std::uint64_t dirAxesHash = 0;
        std::uint64_t dirHash = 0;
        std::uint64_t densityBoundsHash = 0;
    };

    FilmPayloadInput film_payload_input(
        const FilmRawRecipe& filmRaw,
        const FilmDevelopRecipe& filmDevelop,
        const DirCouplersRecipe& dirCouplers,
        const DensityBoundsRecipe& densityBounds);

    bool pack_film_payloads(
        const FilmPayloadInput& input,
        const FilmPreparedView& prepared,
        const float* autoExposureScaleDevice,
        float routeCorrectionScale,
        FilmPayloadPack& out,
        std::string& diagnostic);

    bool pack_film_payloads(
        const FilmRawRecipe& filmRaw,
        const FilmDevelopRecipe& filmDevelop,
        const DirCouplersRecipe& dirCouplers,
        const DensityBoundsRecipe& densityBounds,
        const FilmPreparedView& prepared,
        const float* autoExposureScaleDevice,
        float routeCorrectionScale,
        FilmPayloadPack& out,
        std::string& diagnostic);

    struct VisualGrainStaticAssetsView {
        const std::uint8_t* stbn = nullptr;
        int stbnWidth = 0;
        int stbnHeight = 0;
        int stbnFrames = 0;
        const std::uint8_t* wangTiles = nullptr;
        const std::uint8_t* wangLut = nullptr;
        int wangWidth = 0;
        int wangHeight = 0;
        int wangCount = 0;
        int wangColors = 0;
    };

    struct VisualGrainPreparedGaussianView {
        const float* weights = nullptr;
        int radius = 0;
        float sigma = 0.0f;
        std::uint64_t descriptorHash = 0;
        bool active = false;
    };

    struct VisualGrainPreparedDensityLayersView {
        DeviceCurveView baseCurvesCmy[3] = {};
        const float* curves[3][3] = {
            {nullptr, nullptr, nullptr},
            {nullptr, nullptr, nullptr},
            {nullptr, nullptr, nullptr}};
        std::uint64_t hash = 0;
        bool active = false;
    };

    struct PreparedVisualGrainView {
        VisualGrainStaticAssetsView staticNoise{};
        std::array<VisualGrainPreparedGaussianView, 3> correlation{};
        VisualGrainPreparedGaussianView dyeCloud[3][3] = {};
        VisualGrainPreparedDensityLayersView densityLayers{};
        const Spektrafilm::VisualGrainFrameDescriptor* descriptor = nullptr;
        bool active = false;
    };

    bool pack_visual_grain_payload(
        const Spektrafilm::VisualGrainRecipe& recipe,
        const PreparedVisualGrainView& prepared,
        GrainPayload& outGrain,
        GrainKernelPayload& outKernels,
        std::string& diagnostic);

    bool pack_film_juicer_effects_payload(
        const Spektrafilm::FilmJuicerEffectsFrameDescriptor& descriptor,
        FilmDefectsPayload& outDefects,
        GateWeavePayload& outWeave,
        std::string& diagnostic);

} // namespace JuicerCuda
