// Scanner.h
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "Hash.h"
#include "OutputColor.h"
#include "juicer_cuda_api.h"
#include "ScanRoute.h"
#include "SpectralData.h"


namespace Scanner {

    enum class ScannerMedium : std::uint8_t {
        Negative = 0,
        Print = 1
    };

    struct ScannerIlluminant {
        Spectral::Curve curve;
        float normalization = 0.0f;
        float whiteXYZ[3]{0.0f, 0.0f, 0.0f};
        float whiteXY[2]{0.0f, 0.0f};
        std::uint64_t hash = 0;
    };

    struct ColorRuntime {
        float cat02[9]{0.0f};
        float xyzToRgb[9]{0.0f};
        float illuminantXYZ[3]{0.0f, 0.0f, 0.0f};
        OutputEncoding::Params encoding{};
        // Pointer-free encoding coefficients resolved in Rust.
        FjOutputEncoding preparedEncoding{};
        std::uint64_t outputGamutRecipeHash = 0;
        std::uint64_t hash = 0;
    };

    struct SpectralTablesView {
        std::span<const float> epsC;
        std::span<const float> epsM;
        std::span<const float> epsY;
        std::span<const float> Ax;
        std::span<const float> Ay;
        std::span<const float> Az;
        std::span<const float> baseDensityMin;
        float invYn = 1.0f;
    };

    struct ScannerColorCorrectionDescriptor {
        Spektrafilm::ScanRoute route = Spektrafilm::kDefaultScanRoute;
        bool active = false;
        bool blackCorrection = false;
        bool whiteCorrection = false;
        float targetBlackLinear = 0.0f;
        float targetWhiteLinear = 1.0f;
        float referenceBlackY = 0.0f;
        float referenceWhiteY = 1.0f;
        float xyzSlope = 1.0f;
        float xyzOffset = 0.0f;
        float exposureScale = 1.0f;
        std::uint32_t schemaVersion = 1;
        std::uint64_t hash = 0;
    };

    // Rust resolves post effects and constructs Gaussian weights. Native Root owns CUDA residency.
    struct ScannerPostEffectsDescriptor {
        Spektrafilm::ScanRoute route = Spektrafilm::kDefaultScanRoute;
        bool glareActive = false;
        float glarePercent = 0.0f;
        float glareRoughness = 0.0f;
        float glareBlurSigmaPx = 0.0f;
        int glareBlurRadius = 0;
        float lensBlurSigmaPx = 0.0f;
        int lensBlurRadius = 0;
        float unsharpSigmaPx = 0.0f;
        int unsharpRadius = 0;
        float unsharpAmount = 0.0f;
        std::uint32_t schemaVersion = 2;
        std::uint64_t hash = 0;

        bool active() const noexcept {
            return hash != 0;
        }
    };


} // namespace Scanner
