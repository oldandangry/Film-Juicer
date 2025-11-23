#include "Scanner.h"

#include <algorithm>
#include <cmath>

#include "WorkingState.h"

namespace Scanner {

    void simulate_scanner(
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

