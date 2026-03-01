// Scanner.h
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <vector>

#include "Hash.h"
#include "ProfileJSONLoader.h"
#include "SpectralTypes.h"
#include "OutputEncoding.h"

namespace Spectral {
    struct SpectralTables;
}

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

    struct DensityBuffer {
        // CMY dye densities in SoA form; `medium` tags whether the slab holds negative or print data.
        ScannerMedium medium = ScannerMedium::Negative;
        std::vector<float> c; // cyan dye density (from red layer)
        std::vector<float> m; // magenta dye density (from green layer)
        std::vector<float> y; // yellow dye density (from blue layer)
        int originX = 0;
        int originY = 0;
        int width = 0;
        int height = 0;
        std::ptrdiff_t stride = 0;
    };

    using ScannerDensityBuffer = DensityBuffer;

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
        std::vector<double> cpu;
        std::uint64_t hash = 0;
        std::uint32_t res = 0;
        bool valid = false;
    };

    struct ColorRuntime {
        float cat02[9]{ 0.0f };
        float xyzToRgb[9]{ 0.0f };
        float illuminantXYZ[3]{ 0.0f, 0.0f, 0.0f };
        OutputEncoding::Params encoding{};
        std::uint64_t hash = 0;
    };

    struct ScannerMediumRuntime {
        ScannerMedium medium = ScannerMedium::Negative;
        const Spectral::SpectralTables* tables = nullptr;
        ScannerDensityRange range;
        ScannerIlluminant illuminant;
        Profiles::ProfileGlare glare;
        const ColorRuntime* color = nullptr;
        ScannerStaticKey staticKey;
    };

    inline std::uint64_t hash_glare(const Profiles::ProfileGlare& glare) {
        const float floats[] = {
            glare.percent,
            glare.roughness,
            glare.blur,
            glare.compensationRemovalFactor,
            glare.compensationRemovalDensity,
            glare.compensationRemovalTransition
        };
        for (float v : floats) {
            if (!std::isfinite(v)) {
                return 0;
            }
        }
        const std::uint64_t activeHash = Hash::hash_bytes(&glare.active, sizeof(glare.active));
        const std::uint64_t paramsHash = Hash::hash_float_span(
            floats, sizeof(floats) / sizeof(floats[0]));
        const std::uint64_t fields[] = { activeHash, paramsHash };
        return Hash::hash_bytes(fields, sizeof(fields));
    }

    inline void finalize_static_key(ScannerStaticKey& key) {
        const std::uint64_t fields[] = {
            static_cast<std::uint64_t>(key.medium),
            key.tablesHash,
            key.densityRangeHash,
            key.colorRuntimeHash,
            static_cast<std::uint64_t>(key.lutResolution)
        };
        key.hash = Hash::hash_bytes(fields, sizeof(fields));
    }

    inline std::uint64_t identity_color_runtime_hash() {
        static const std::uint64_t h =
            Hash::hash_bytes("identity_color_runtime", sizeof("identity_color_runtime") - 1);
        return h;
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

} // namespace Scanner
