#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

#include "ColorTransforms.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"
#include "OutputColor.h"
#include "ProfileCatalog.h"
#include "RenderControls.h"
#include "ScanRoute.h"
#include "ScatterHalation.h"
#include "SpectralData.h"

// SF_TEMP_BRIDGE_CAPTURED_INPUT_IDENTITY: native capture/conversion/hash through S6.
struct ParamSnapshot {
    double printExposure = 1.0;
    double printPreflashExposure = 0.0;
    double printGammaFactor = 1.0;
    double preflashMFilterCc = 0.0;
    double preflashYFilterCc = 0.0;
    double printShadowCompensationFactor = 0.0;
    double printShadowCompensationDensity = 1.2;
    double printShadowCompensationTransition = 0.3;
    double glarePercent = 0.03;
    double glareRoughness = 0.7;
    double glareBlurSigmaPx = 0.5;
    double cameraExposureCompensationEv = 0.0;
    float cameraFilmFormatLongEdgeMm = 35.0f;
    float filmGammaFactor = 1.0f;
    double scannerLensBlurSigmaPx = 0.0;
    double scannerBlackLevel = 0.01;
    double scannerWhiteLevel = 0.98;
    double gateWeaveAmount = 0.0;

    std::array<double, 2> scannerUnsharpMask{{0.7, 0.7}};
    std::array<double, 3> printUiYmcCc{};
    std::array<double, 3> cameraFilterUV{{1.0, 410.0, 8.0}};
    std::array<double, 3> cameraFilterIR{{1.0, 675.0, 15.0}};

    Spektrafilm::DiffusionFilterAuthoredControls cameraDiffusion;
    Spektrafilm::DiffusionFilterAuthoredControls enlargerDiffusion;
    Spektrafilm::DirCouplersControls dirCouplers;

    std::string filmProfileKey = Spektrafilm::kDefaultFilmProfileKey;
    std::string printProfileKey = Spektrafilm::kDefaultPrintProfileKey;

    int spectralUpsamplingMode = 0;
    int refIll = 0;
    int enlIll = 3;
    int normalizePrintExposure = 1;
    int printExposureCompensation = 1;
    int inputColorSpace = Spectral::inputColorSpaceToIndex(Spectral::InputColorSpace::DaVinciWideGamut);
    int inputCctfDecoding = 0;
    int inputCompressionEnabled = 1;
    int hanatos2025AdaptationWindow = 1;
    int hanatos2025AdaptationSurface = 0;
    int cameraAutoExposureEnabled = 1;
    int cameraMeteringMethod = 0;
    ScatterHalationControls scatterHalationControls;
    int scannerBlackCorrection = 0;
    int scannerWhiteCorrection = 0;
    int scannerLutResolution = 17;
    int outputColorSpace = OutputEncoding::toIndex(OutputEncoding::ColorSpace::sRGB);
    int outputCctfEncoding = 1;
    int outputGamutCompressionEnabled = 1;
    Spektrafilm::VisualGrainControls grainControls;
    float filmDustAmount = 0.0f;
    float filmScratchAmount = 0.0f;
    float gateDustAmount = 0.0f;
    float gateScratchAmount = 0.0f;
    Spektrafilm::ScanRoute scanRoute = Spektrafilm::kDefaultScanRoute;

    bool glareActive = true;
    bool cameraFilterOverride = false;
};

uint64_t hash_params(const ParamSnapshot& p);
bool set_gamma_snapshot_values(
    double authoredFilmGammaFactor,
    double authoredPrintGammaFactor,
    ParamSnapshot& snapshot,
    std::string& outDiagnostic);

// SF_TEMP_BRIDGE_NATIVE_SUBMISSION_STATE: temporal/submission ownership moves at S5.D.
struct NativeSubmissionState {
    std::uint64_t sessionSeed = 0;
    std::uint64_t instanceToken = 0;
    std::atomic<std::uint64_t> submissionSnapshotIdNext{1};

    // Snapshot latch: all submissions for the same frame token reuse one immutable snapshot payload.
    std::mutex submissionSnapshotLatchMutex;
    bool submissionSnapshotLatchValid = false;
    JuicerCuda::ResourceManager::SubmissionSnapshot submissionSnapshotLatch{};
};

bool spektrafilm_profile_catalog_ready();
const char* spektrafilm_profile_catalog_failure();
int film_profile_option_count();
const char* film_profile_option_key(int index);
const char* film_profile_option_label(int index);
int print_profile_option_count();
const char* print_profile_option_key(int index);
const char* print_profile_option_label(int index);
