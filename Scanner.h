// Scanner.h
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "Hash.h"
#include "SpectralProcessing.h"
#include "ColorTransforms.h"
#include "Couplers.h"
#include "ParamNames.h"
#include "WorkingState.h"

// Forward-declare WorkingState so we don't create header cycles
struct WorkingState;

namespace Scanner {

    enum class ScannerMedium : int {
        Negative = 0,
        Print = 1
    };

    struct Options {
        float lensBlurSigmaPx = 0.55f;
        float unsharpSigmaPx = 0.7f;
        float unsharpAmount = 1.0f;
    };

    struct Settings {
        bool useLut = true;
        std::uint32_t lutResolution = 17;
    };

    struct ScannerDensityRange {
        float min_cmy[3]{ 0.0f, 0.0f, 0.0f };
        float max_cmy[3]{ 0.0f, 0.0f, 0.0f };
        float inv_max_cmy[3]{ 0.0f, 0.0f, 0.0f };
        std::uint64_t digest = 0;
    };

    struct ScannerStaticKey {
        ScannerMedium medium = ScannerMedium::Negative;
        std::uint64_t tablesHash = 0;
        std::uint64_t densityRangeHash = 0;
        std::uint64_t glareHash = 0;
        std::uint64_t colorRuntimeHash = 0;
        std::uint32_t lutResolution = 0;
        std::uint64_t hash = 0;
    };

    struct ScannerRuntimeKey {
        std::uint64_t settingsHash = 0;
        std::uint32_t frameBoundsVersion = 0;
        std::uint64_t hash = 0;
    };

    struct ScannerKey {
        ScannerStaticKey staticKey;
        ScannerRuntimeKey runtimeKey;
        std::uint64_t hash = 0;
    };

    struct SpectralLutBuffer {
        std::vector<float> cpu;
        std::uint64_t hash = 0;
        std::uint32_t res = 0;
        bool valid = false;
    };

    inline void finalize_static_key(ScannerStaticKey& key) {
        const std::uint64_t fields[] = {
            static_cast<std::uint64_t>(key.medium),
            key.tablesHash,
            key.densityRangeHash,
            key.glareHash,
            key.colorRuntimeHash,
            static_cast<std::uint64_t>(key.lutResolution)
        };
        key.hash = Hash::hash_bytes(fields, sizeof(fields));
    }

    inline void finalize_runtime_key(ScannerRuntimeKey& key) {
        const std::uint64_t fields[] = {
            key.settingsHash,
            static_cast<std::uint64_t>(key.frameBoundsVersion)
        };
        key.hash = Hash::hash_bytes(fields, sizeof(fields));
    }

    inline void finalize_scanner_key(ScannerKey& key) {
        finalize_static_key(key.staticKey);
        finalize_runtime_key(key.runtimeKey);
        const std::uint64_t fields[] = { key.staticKey.hash, key.runtimeKey.hash };
        key.hash = Hash::hash_bytes(fields, sizeof(fields));
    }

    // Legacy compatibility shim (to be removed in later stages).
    struct Params {
        bool enabled = true;
    };

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
