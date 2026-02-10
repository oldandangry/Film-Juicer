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
    std::array<std::array<std::vector<float>, 3>, 3> densityCurvesLayers{}; // [layer][channel] values on density axis
    bool hasDensityCurvesLayers = false;
    Profiles::GrainMetadata grain;
    Profiles::HalationMetadata halation;
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
    Scanner::ColorRuntime negativeColorRuntime;
    Scanner::ScannerMediumRuntime negativeMediumRuntime;

    // Per-instance scanner metadata for print viewing path
    Scanner::ScannerIlluminant printScannerIlluminant;
    Scanner::ScannerDensityRange printDensityRange;
    Scanner::ScannerStaticKey printStaticKey;
    Scanner::ColorRuntime printColorRuntime;
    Scanner::ScannerMediumRuntime printMediumRuntime;

    // Validity flags for scanner media
    bool negativeScannerValid = false;
    bool printScannerValid = false;
    bool printGlareCompensated = false;

    // SPD reconstruction per-instance (non-global)
    float spdSInv[9] = { 1,0,0, 0,1,0, 0,0,1 };
    bool  spdReady = false;

    // Film raw conversion configuration (input colour space, normalization)
    Spectral::FilmRawConfig filmRaw;

    // Snapshot of print runtime used for this WorkingState build (immutable during render)
    std::shared_ptr<const Print::Runtime> printRT;

    Spectral::NegativeCouplerParams negParams{};

    // Versioning for atomic swap / debugging
    std::uint64_t fullHash = 0;
    std::uint64_t uploadCoreHash = 0;
    std::uint64_t coreHash = 0;
    std::uint64_t dirHash = 0;
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
    auto sample_layers = [&](const Spectral::Curve& cB,
        const Spectral::Curve& cG,
        const Spectral::Curve& cR,
        const float le[3],
        float layerD_out[3]) {
            layerD_out[0] = Spectral::sample_density_at_logE(cB, le[0], ws.gammaFactorB); // Yellow (blue layer)
            layerD_out[1] = Spectral::sample_density_at_logE(cG, le[1], ws.gammaFactorG); // Magenta (green layer)
            layerD_out[2] = Spectral::sample_density_at_logE(cR, le[2], ws.gammaFactorR); // Cyan (red layer)
        };

    auto write_cmy = [](const float layerD[3], float D_out_local[3]) {
        D_out_local[0] = layerD[2]; // C from red layer
        D_out_local[1] = layerD[1]; // M from green layer
        D_out_local[2] = layerD[0]; // Y from blue layer
    };

    const Spectral::Curve& precorrectedB = ws.dirPrecorrected ? ws.dirDensB : ws.densB;
    const Spectral::Curve& precorrectedG = ws.dirPrecorrected ? ws.dirDensG : ws.densG;
    const Spectral::Curve& precorrectedR = ws.dirPrecorrected ? ws.dirDensR : ws.densR;

#ifdef JUICER_ENABLE_COUPLERS
    if (mode == DirSampleMode::ApplyRuntime && dirRT.active) {
        float layerPre[3];
        sample_layers(ws.densB, ws.densG, ws.densR, logE, layerPre);
        Couplers::ApplyInputLogE io{ {logE[0], logE[1], logE[2]}, {layerPre[0], layerPre[1], layerPre[2]} };
        Couplers::apply_runtime_logE_with_curves(io, dirRT, ws.densB, ws.densG, ws.densR);
        float layerPost[3];
        sample_layers(precorrectedB, precorrectedG, precorrectedR, io.logE, layerPost);
        write_cmy(layerPost, D_out);
        return;
    }
#else
    (void)dirRT;
    (void)mode;
#endif

    float layerD[3];
    sample_layers(ws.densB, ws.densG, ws.densR, logE, layerD);
    write_cmy(layerD, D_out);
}
