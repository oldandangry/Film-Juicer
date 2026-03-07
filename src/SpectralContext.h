#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>

#include "SpectralTypes.h"
#include "NpyLoader.h"

namespace Spectral {

    enum class SpectralMutationStage : std::uint8_t {
        None = 0,
        Bootstrap = 1,
        Rebuild = 2
    };

    const char* to_cstr(SpectralMutationStage stage) noexcept;

    class SpectralMutationScope {
    public:
        SpectralMutationScope(SpectralMutationStage stage, const char* owner = nullptr) noexcept;
        ~SpectralMutationScope() noexcept;

        bool active() const noexcept { return _active; }

    private:
        SpectralMutationStage _stage = SpectralMutationStage::None;
        const char* _owner = nullptr;
        bool _active = false;
    };

    bool spectral_mutation_scope_active() noexcept;
    SpectralMutationStage spectral_mutation_stage() noexcept;
    bool require_spectral_mutation_scope(const char* action) noexcept;

    struct SpectralContext {
        PrecomputeStatus precomputeStatus;
        std::mutex precomputeMutex;
        int lastIllumChoice = -1;

        SpectralShape shape;
        std::vector<float> epsYTable;
        std::vector<float> epsMTable;
        std::vector<float> epsCTable;
        std::vector<float> xbarTable;
        std::vector<float> ybarTable;
        std::vector<float> zbarTable;
        std::vector<float> baselineMinTable;
        std::vector<float> baselineMidTable;
        std::vector<float> illumTable;
        std::vector<float> Ax;
        std::vector<float> Ay;
        std::vector<float> Az;
        std::vector<float> lambda;

        float ynNorm = 1.0f;
        float invYn = 1.0f;
        float deltaLambda = kDelta;

        Curve illuminantCurve;
        Curve sensBlue, sensGreen, sensRed;
        Curve densityCurveB, densityCurveG, densityCurveR;
        Curve epsY, epsM, epsC;
        Curve xBar, yBar, zBar;
        Curve baseMin, baseMid;
        bool hasBaseline = false;

        std::atomic<bool> hanatosAvailable{ false };
        NpySpectraLUT hanSpectra;
        std::atomic<bool> mallettAvailable{ false };
        NpyFloat2D mallettBasis;

        std::atomic<bool> spdInit{ false };
        float sInv[9] = { 1.0f,0.0f,0.0f, 0.0f,1.0f,0.0f, 0.0f,0.0f,1.0f };
    };

    SpectralContext& context();

    void mark_mixing_dirty();
    void mark_spectral_tables_dirty();
    void increment_illum_version();
    void increment_shape_version();

} // namespace Spectral
