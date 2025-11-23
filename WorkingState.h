// WorkingState.h
#pragma once

#include <cstdint>
#include <memory>
#include "SpectralData.h"
#include "SpectralProcessing.h"
#include "FilmProcessing.h"
#include "Couplers.h"
#include "ProfileJSONLoader.h"
#include "Scanner.h"

// Forward declaration
namespace Print {
    struct Runtime;
}

// Per‑instance, derived state used for rendering.
// Built from BaseState in rebuild_working_state() and never mutated in render().
struct WorkingState {
    // Density curves (after rebuild transforms)
    Spectral::Curve densB;
    Spectral::Curve densG;
    Spectral::Curve densR;
    Profiles::GrainMetadata grain;
    Profiles::ProfileGlare negativeGlare;
    Profiles::ProfileGlare printGlare;

    // Sensitivity curves from profile (per agx-emulsion parity: pre-balanced, not runtime balanced)
    Spectral::Curve sensB;
    Spectral::Curve sensG;
    Spectral::Curve sensR;

    // DEPRECATED: No longer used. Per agx-emulsion parity, profiles contain pre-balanced
    // sensitivities; runtime rebalancing was removed. Always empty. See claude-review.md.
    Spectral::Curve negSensB;
    Spectral::Curve negSensG;
    Spectral::Curve negSensR;

    // Baseline curves and flag
    Spectral::Curve baseMin;
    Spectral::Curve baseMid;
    bool hasBaseline = false;
    float baselineMixReference = 0.0f;
    float printBaselineMixReference = 0.0f;

    // Per-channel gamma factors for density curve interpolation (matches agx-emulsion parity)
    float gammaFactorB = 1.0f;
    float gammaFactorG = 1.0f;
    float gammaFactorR = 1.0f;

    // DIR runtime (per-pixel)
    Couplers::Runtime dirRT;

    // Density curves used after per-pixel DIR corrections (pre-DIR characteristic curves)
    Spectral::Curve dirDensB;
    Spectral::Curve dirDensG;
    Spectral::Curve dirDensR;

    // Whether DIR pre-correction was applied to density curves in this build
    bool dirPrecorrected = false;

    // Normalization constants (frozen from pre-DIR balanced curves)
    float dMax[3] = { 1.0f, 1.0f, 1.0f };

    // Per-instance precomputed spectral tables for the viewing illuminant
    Spectral::SpectralTables tablesView;

    // Per-instance spectral tables for the print paper viewing path
    Spectral::SpectralTables tablesPrint;

    // Per-instance spectral tables for the film reference illuminant (SPD reconstruction)
    Spectral::SpectralTables tablesRef;

    // Per-instance spectral tables for scanner/negative viewing (D50 parity)
    Spectral::SpectralTables tablesScan;
    Scanner::ScannerIlluminant negativeScannerIlluminant;
    Scanner::ScannerDensityRange negativeDensityRange;
    Scanner::ScannerStaticKey negativeStaticKey;

    // Per-instance scanner metadata for print viewing path
    Scanner::ScannerIlluminant printScannerIlluminant;
    Scanner::ScannerDensityRange printDensityRange;
    Scanner::ScannerStaticKey printStaticKey;

    // SPD reconstruction per-instance (non-global)
    float spdSInv[9] = { 1,0,0, 0,1,0, 0,0,1 };
    bool  spdReady = false;

    // Film raw conversion configuration (input colour space, normalization)
    Spectral::FilmRawConfig filmRaw;

    // Snapshot of print runtime used for this WorkingState build (immutable during render)
    std::unique_ptr<Print::Runtime> printRT;

    Spectral::NegativeCouplerParams negParams{};

    // Versioning for atomic swap / debugging
    std::uint64_t buildCounter = 0;
};

enum class DirSampleMode {
    ApplyRuntime,
    BypassRuntime
};

inline void sample_negative_densities(
    const WorkingState& ws,
    const Couplers::Runtime& dirRT,
    const float logE[3],
    float D_out[3],
    DirSampleMode mode = DirSampleMode::ApplyRuntime)
{
    auto sample_with_curves = [&](const Spectral::Curve& cB,
        const Spectral::Curve& cG,
        const Spectral::Curve& cR,
        const float le[3],
        float D_out_local[3]) {
            D_out_local[0] = Spectral::sample_density_at_logE(cB, le[0], ws.gammaFactorB);
            D_out_local[1] = Spectral::sample_density_at_logE(cG, le[1], ws.gammaFactorG);
            D_out_local[2] = Spectral::sample_density_at_logE(cR, le[2], ws.gammaFactorR);
        };

    const Spectral::Curve& precorrectedB = ws.dirPrecorrected ? ws.dirDensB : ws.densB;
    const Spectral::Curve& precorrectedG = ws.dirPrecorrected ? ws.dirDensG : ws.densG;
    const Spectral::Curve& precorrectedR = ws.dirPrecorrected ? ws.dirDensR : ws.densR;

#ifdef JUICER_ENABLE_COUPLERS
    if (mode == DirSampleMode::ApplyRuntime && dirRT.active) {
        float D_pre[3];
        sample_with_curves(ws.densB, ws.densG, ws.densR, logE, D_pre);
        Couplers::ApplyInputLogE io{ {logE[0], logE[1], logE[2]}, {D_pre[0], D_pre[1], D_pre[2]} };
        Couplers::apply_runtime_logE_with_curves(io, dirRT, ws.densB, ws.densG, ws.densR);
        sample_with_curves(precorrectedB, precorrectedG, precorrectedR, io.logE, D_out);
        return;
    }
#else
    (void)dirRT;
    (void)mode;
#endif

    sample_with_curves(ws.densB, ws.densG, ws.densR, logE, D_out);
}
