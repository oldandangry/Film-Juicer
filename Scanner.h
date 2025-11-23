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

    struct ScannerIlluminant {
        Spectral::Curve curve;
        float normalization = 0.0f;
        float whiteXYZ[3]{ 0.0f, 0.0f, 0.0f };
        float whiteXY[2]{ 0.0f, 0.0f };
        std::uint64_t hash = 0;
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

    void simulate_scanner(
        const float rgbIn[3],
        float rgbOut[3],
        const Scanner::Params& scannerParams,
        const Couplers::Runtime& dirRT,
        const WorkingState& ws,
        float exposureScale);

} // namespace Scanner
