// Scanner.h
#pragma once

#include <algorithm>
#include <cmath>

#include "SpectralProcessing.h"
#include "ColorTransforms.h"
#include "Couplers.h"
#include "ParamNames.h"
#include "WorkingState.h"

// Forward-declare WorkingState so we don't create header cycles
struct WorkingState;

namespace Scanner {

    struct Params {
        bool enabled = false;
        bool autoExposure = true;
        float targetY = 0.18f;
        float filmLongEdgeMm = 36.0f;
    };

    inline void fetch_params(OfxParameterSuiteV1* paramSuite, OfxParamSetHandle paramSet, Params& out) {
        if (!paramSuite || !paramSet) { out = {}; out.autoExposure = true; out.targetY = 0.18f; return; }
        OfxParamHandle hEnabled = nullptr, hAutoExp = nullptr, hTargetY = nullptr, hFilmLongEdge = nullptr;
        paramSuite->paramGetHandle(paramSet, "ScannerEnabled", &hEnabled, nullptr);
        paramSuite->paramGetHandle(paramSet, "ScannerAutoExposure", &hAutoExp, nullptr);
        paramSuite->paramGetHandle(paramSet, "ScannerTargetY", &hTargetY, nullptr);
        paramSuite->paramGetHandle(paramSet, JuicerParams::kScannerFilmLongEdgeMm, &hFilmLongEdge, nullptr);
        int enabled = 0, autoExp = 1;
        double targetY = 0.18;
        double filmLongEdge = 36.0;
        if (hEnabled) paramSuite->paramGetValue(hEnabled, &enabled);
        if (hAutoExp) paramSuite->paramGetValue(hAutoExp, &autoExp);
        if (hTargetY) paramSuite->paramGetValue(hTargetY, &targetY);
        if (hFilmLongEdge) paramSuite->paramGetValue(hFilmLongEdge, &filmLongEdge);
        out.enabled = (enabled != 0);
        out.autoExposure = (autoExp != 0);
        out.targetY = static_cast<float>(targetY);
        out.filmLongEdgeMm = (std::isfinite(filmLongEdge) && filmLongEdge > 0.0)
            ? static_cast<float>(filmLongEdge)
            : 36.0f;
    }

    inline void simulate_scanner(
        const float rgbIn[3],
        float rgbOut[3],
        const Scanner::Params& scannerParams,
        const Couplers::Runtime& dirRT,
        const WorkingState& ws,
        float exposureScale)
    {
        const float exposureScaleSafe =
            (std::isfinite(exposureScale) && exposureScale > 0.0f)
            ? exposureScale
            : 1.0f;

        if (!scannerParams.enabled) {
            rgbOut[0] = rgbIn[0];
            rgbOut[1] = rgbIn[1];
            rgbOut[2] = rgbIn[2];
            return;
        }

        // DWG → layer exposures (per-instance SPD vs Matrix, no global toggle)
        float E[3];
        const Spectral::SpectralTables* tablesSPD =
            (ws.spdReady && ws.tablesRef.K > 0) ? &ws.tablesRef : nullptr;
        // Per agx-emulsion parity: use same sensitivities everywhere.
        Spectral::rgb_input_to_film_raw(
            rgbIn, E, exposureScaleSafe,
            ws.filmRaw,
            tablesSPD,
            (ws.spdReady ? ws.spdSInv : nullptr),
            ws.spdReady,
            ws.sensB, ws.sensG, ws.sensR);

        // Log exposure domain for density lookup (offsets already baked into curves)
        float logE[3] = {
            std::log10(std::max(0.0f, E[0]) + 1e-10f),
            std::log10(std::max(0.0f, E[1]) + 1e-10f),
            std::log10(std::max(0.0f, E[2]) + 1e-10f)
        };

        float D_cmy[3];
        sample_negative_densities(ws, dirRT, logE, D_cmy);

        // Spectral integration using scanner viewing tables (fallback to view tables if missing)
        const Spectral::SpectralTables* tables = nullptr;
        if (ws.tablesScan.K > 0) {
            tables = &ws.tablesScan;
        }
        else if (ws.tablesView.K > 0) {
            tables = &ws.tablesView;
        }

        if (!tables) {
            rgbOut[0] = rgbIn[0];
            rgbOut[1] = rgbIn[1];
            rgbOut[2] = rgbIn[2];
            return;
        }

        float XYZ[3] = { 0.0f, 0.0f, 0.0f };
        if (ws.hasBaseline && tables->hasBaseline) {
            Spectral::dyes_to_XYZ_with_baseline_given_tables(*tables, D_cmy, XYZ);
        }
        else {
            Spectral::dyes_to_XYZ_given_tables(*tables, D_cmy, XYZ);
        }

        // XYZ → DWG with chromatic adaptation based on the scan illuminant
        Spectral::XYZ_to_DWG_linear_adapted(*tables, XYZ, rgbOut);
        rgbOut[0] = std::max(0.0f, rgbOut[0]);
        rgbOut[1] = std::max(0.0f, rgbOut[1]);
        rgbOut[2] = std::max(0.0f, rgbOut[2]);
    }

} // namespace Scanner
