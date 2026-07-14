#pragma once

#include <atomic>
#include <array>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "ProcessRoot.h"
#include "Print.h"
#include "ProfileCatalog.h"
#include "ProfileJSONLoader.h"
#include "RenderRecipe.h"
#include "ScanRoute.h"
#include "ColorTransforms.h"
#include "FilmProcessing.h"
#include "Couplers.h"
#include "Scanner.h"
#include "SpectralData.h"
#include "ofxImageEffect.h"

namespace ScannerOptics {
    Scanner::ColorRuntime build_color_runtime(
        const Scanner::ScannerMediumRuntime& medium,
        const OutputEncoding::Params& outputEncoding);

} // namespace ScannerOptics

namespace JuicerAtomic {

    template <typename T>
    inline std::shared_ptr<T> load_shared_ptr(const std::shared_ptr<T>* p) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
        std::shared_ptr<T> v = std::atomic_load(p);
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
        return v;
    }

    template <typename T>
    inline void store_shared_ptr(std::shared_ptr<T>* p, std::shared_ptr<T> value) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
        std::atomic_store(p, std::move(value));
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    }

} // namespace JuicerAtomic

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"

#endif

inline std::filesystem::path data_dir_path() {
    std::filesystem::path path(JuicerProcess::root().data_dir());
    path.make_preferred();
    return path;
}

template <typename... Parts>
inline std::filesystem::path data_dir_path(Parts&&... parts) {
    std::filesystem::path path(JuicerProcess::root().data_dir());
    (void)std::initializer_list<int>{
        ((path /= std::filesystem::path(std::forward<Parts>(parts))), 0)...};
    path.make_preferred();
    return path;
}

inline std::string ensure_trailing_separator(std::string value) {
    const char sep = std::filesystem::path::preferred_separator;
    if (!value.empty() && value.back() != sep) {
        value.push_back(sep);
    }
    return value;
}

template <typename... Parts>
inline std::string data_dir_string(Parts&&... parts) {
    return data_dir_path(std::forward<Parts>(parts)...).string();
}

struct BaseState {
    Spectral::Curve epsY, epsM, epsC;
    Spectral::Curve sensB, sensG, sensR;
    Spectral::Curve densB, densG, densR;
    Spectral::Curve baseDensityMin, baseDensityMid;
    std::array<std::array<std::vector<float>, 3>, 3> densityCurvesLayers{}; // [layer][channel] values on LOG_EXPOSURE axis
    bool hasDensityCurvesLayers = false;
    float dyeDensityMinFactor = 1.0f;
    std::array<float, 3> gammaFactor{{1.0f, 1.0f, 1.0f}};
    bool hasBaseline = false;
    std::string referenceIlluminant;
    std::string viewingIlluminant;
    std::vector<float> densityMidNeutral;
    std::vector<float> logExposureMidNeutral;
    Profiles::DirProfile dirCouplers;
    Profiles::MaskingCouplersProfile maskingCouplers;
    std::array<float, 3> cameraFilterUV{{1.0f, 410.0f, 8.0f}};
    std::array<float, 3> cameraFilterIR{{1.0f, 675.0f, 15.0f}};
    bool cameraFilterDefined = false;
    Profiles::GrainMetadata grain;
    Profiles::HalationMetadata halation;
    Profiles::ProfileGlare glare;
};

struct DirectRenderPayload {
    Spectral::SpectralTables exposureTables;
    std::array<float, 9> spdSInv{{1, 0, 0, 0, 1, 0, 0, 0, 1}};
    Spectral::FilmRawConfig filmRawConfig;
    Spectral::SpectralTables scannerTables;
    Scanner::ColorRuntime scannerColor;
    std::uint64_t uploadCoreHash = 0;
    std::uint64_t scannerHash = 0;
};

struct DirectRenderState {
    Spektrafilm::RenderRecipe recipe;
    DirectRenderPayload payload;
    std::uint64_t buildCounter = 0;
};

struct PrintRenderPayload {
    Spectral::SpectralTables exposureTables;
    std::array<float, 9> spdSInv{{1, 0, 0, 0, 1, 0, 0, 0, 1}};
    Spectral::FilmRawConfig filmRawConfig;
    Spectral::SpectralTables scannerTables;
    Scanner::ColorRuntime scannerColor;
    std::uint64_t uploadCoreHash = 0;
    std::uint64_t scannerHash = 0;
};

struct PrintRenderState {
    Spektrafilm::RenderRecipe recipe;
    PrintRenderPayload payload;
    std::uint64_t buildCounter = 0;
};

// Per-instance, derived state used for rendering.
// Built from BaseState in rebuild_working_state() and never mutated in render().
struct WorkingState {
    Spektrafilm::RenderRecipe recipe;

    Spectral::Curve densB;
    Spectral::Curve densG;
    Spectral::Curve densR;
    std::array<std::array<std::vector<float>, 3>, 3> densityCurvesLayers{};
    bool hasDensityCurvesLayers = false;
    Profiles::GrainMetadata grain;
    Profiles::HalationMetadata halation;
    Profiles::ProfileGlare negativeGlare;
    Profiles::ProfileGlare printGlare;

    Spectral::Curve sensB;
    Spectral::Curve sensG;
    Spectral::Curve sensR;

    Spectral::Curve negSensB;
    Spectral::Curve negSensG;
    Spectral::Curve negSensR;

    Spectral::Curve baseDensityMin;
    Spectral::Curve baseDensityMid;
    bool hasBaseline = false;
    float densityBaselineMixReference = 0.0f;
    float printBaselineMixReference = 0.0f;

    float gammaFactorB = 1.0f;
    float gammaFactorG = 1.0f;
    float gammaFactorR = 1.0f;

    Couplers::Runtime dirRT;

    Spectral::Curve dirDensB;
    Spectral::Curve dirDensG;
    Spectral::Curve dirDensR;

    bool dirPrecorrected = false;
    float dMax[3] = {1.0f, 1.0f, 1.0f};

    Spectral::SpectralTables tablesView;
    Spectral::SpectralTables tablesPrint;
    Spectral::SpectralTables tablesRef;
    Spectral::SpectralTables tablesScan;
    Scanner::ScannerIlluminant negativeScannerIlluminant;
    Scanner::ScannerDensityRange negativeDensityRange;
    Scanner::ScannerStaticKey negativeStaticKey;
    Scanner::ScannerRuntimeEffectsKey negativeEffectsKey;
    Scanner::ColorRuntime negativeColorRuntime;
    Scanner::ScannerMediumRuntime negativeMediumRuntime;

    Scanner::ScannerIlluminant printScannerIlluminant;
    Scanner::ScannerDensityRange printDensityRange;
    Scanner::ScannerStaticKey printStaticKey;
    Scanner::ScannerRuntimeEffectsKey printEffectsKey;
    Scanner::ColorRuntime printColorRuntime;
    Scanner::ScannerMediumRuntime printMediumRuntime;

    bool negativeScannerValid = false;
    bool printScannerValid = false;
    bool printGlareCompensated = false;

    float spdSInv[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    bool spdReady = false;
    Spectral::FilmRawConfig filmRaw;
    std::shared_ptr<const Print::Runtime> printRT;
    Spectral::NegativeCouplerParams negParams{};

    std::uint64_t fullHash = 0;
    std::uint64_t uploadCoreHash = 0;
    std::uint64_t coreHash = 0;
    std::uint64_t coreShareHash = 0;
    std::uint64_t dirHash = 0;
    std::uint64_t buildCounter = 0;
};

enum class DirSampleMode : std::uint8_t {
    ApplyRuntime,
    BypassRuntime
};

inline void sample_negative_densities(
    const WorkingState& ws,
    const Couplers::Runtime& dirRT,
    const float logE[3],
    float D_out[3],
    DirSampleMode mode = DirSampleMode::ApplyRuntime) {
    auto sample_layers = [&](const Spectral::Curve& cB,
                             const Spectral::Curve& cG,
                             const Spectral::Curve& cR,
                             const float le[3],
                             float layerD_out[3]) {
        layerD_out[0] = Spectral::sample_density_at_logE(cB, le[0], ws.gammaFactorB);
        layerD_out[1] = Spectral::sample_density_at_logE(cG, le[1], ws.gammaFactorG);
        layerD_out[2] = Spectral::sample_density_at_logE(cR, le[2], ws.gammaFactorR);
    };

    auto write_cmy = [](const float layerD[3], float D_out_local[3]) {
        D_out_local[0] = layerD[2];
        D_out_local[1] = layerD[1];
        D_out_local[2] = layerD[0];
    };

    (void)dirRT;
    (void)mode;

    float layerD[3];
    sample_layers(ws.densB, ws.densG, ws.densR, logE, layerD);
    write_cmy(layerD, D_out);
}

struct ParamSnapshot {
    std::uint64_t filmProfileAssetVersionToken = 0;
    std::uint64_t printProfileAssetVersionToken = 0;

    double printExposure = 1.0;
    double printPreflashExposure = 0.0;
    double preflashMFilterCc = 0.0;
    double preflashYFilterCc = 0.0;
    double printShadowCompensationFactor = 0.0;
    double printShadowCompensationDensity = 1.2;
    double printShadowCompensationTransition = 0.3;
    double glarePercent = 0.03;
    double glareRoughness = 0.7;
    double glareBlurSigmaPx = 0.5;
    double printDminFactor = 0.4;
    double couplersAmount = 1.0;
    double couplersInhibitionSameLayer = 1.0;
    double couplersInhibitionInterlayer = 1.0;
    double couplersDiffusionSizeUm = 20.0;
    double cameraExposureCompensationEv = 0.0;
    double cameraFilmFormatLongEdgeMm = 36.0;
    double scannerLensBlurSigmaPx = 0.0;
    double scannerBlackLevel = 0.01;
    double scannerWhiteLevel = 0.98;
    double gateWeaveAmount = 1.0;

    std::array<double, 2> couplersGammaInterlayerRToGb{{0.353, 0.302}};
    std::array<double, 2> couplersGammaInterlayerGToRb{{0.154, 0.353}};
    std::array<double, 2> couplersGammaInterlayerBToRg{{0.168, 0.226}};
    std::array<double, 2> scannerUnsharpMask{{0.7, 0.7}};
    std::array<double, 3> printUiYmcCc{};
    std::array<double, 3> couplersGammaSameLayerRgb{{0.336, 0.319, 0.273}};
    std::array<double, 3> cameraFilterUV{{1.0, 410.0, 8.0}};
    std::array<double, 3> cameraFilterIR{{1.0, 675.0, 15.0}};

    std::string filmProfileKey = Spektrafilm::kDefaultFilmProfileKey;
    std::string printProfileKey = Spektrafilm::kDefaultPrintProfileKey;

    int spectralUpsamplingMode = 0;
    int refIll = 0;
    int enlIll = 3;
    int enlDichroicSet = 0;
    int normalizePrintExposure = 1;
    int printExposureCompensation = 1;
    int couplersActive = 1;
    int couplersGammaUseStock = 1;
    int inputColorSpace = Spectral::inputColorSpaceToIndex(Spectral::InputColorSpace::DaVinciWideGamut);
    int inputCctfDecoding = 0;
    int hanatos2025AdaptationWindow = 1;
    int hanatos2025AdaptationSurface = 0;
    int cameraAutoExposureEnabled = 1;
    int cameraMeteringMethod = 0;
    int exactScatterHalationActive = 0;
    int scannerBlackCorrection = 0;
    int scannerWhiteCorrection = 0;
    int scannerUseLut = 1;
    int scannerLutResolution = 17;
    int outputColorSpace = OutputEncoding::toIndex(OutputEncoding::ColorSpace::sRGB);
    int outputCctfEncoding = 1;
    int outputLinearPassThrough = 0;
    Profiles::GrainMetadata grainControls;
    Spektrafilm::ScanRoute scanRoute = Spektrafilm::kDefaultScanRoute;

    bool directRoutePrintProfileExcluded = false;
    bool directRouteNeutralCalibrationExcluded = false;
    bool glareActive = true;
    bool cameraFilterOverride = false;
};

uint64_t hash_params(const ParamSnapshot& p);
uint64_t hash_params_core(const ParamSnapshot& p);
uint64_t hash_params_dir(const ParamSnapshot& p);

struct IlluminantOverrideFlags {
    bool reference = false;
    bool enlarger = false;
};

struct PendingParamsState {
    // Leaf lock for the coalesced pending-params snapshot. Do not nest with InstanceState::m.
    std::mutex m;
    ParamSnapshot params;
    std::uint64_t fullHash = 0;
    std::uint64_t coreHash = 0;
    std::uint64_t dirHash = 0;
};

struct InstanceState {
    // Lock-order rule for P6: rebuildMutex may precede m during rebuild snapshot/publication.
    // m is otherwise a short publication/snapshot lock and must not be held across heavy rebuild
    // work, sleep/wait paths, or while acquiring the leaf mutexes below.
    std::mutex m;
    // Serializes working-state rebuilds so the heavy rebuild path can run outside InstanceState::m.
    std::mutex rebuildMutex;
    BaseState base;
    // Published render snapshot. Readers use std::atomic_load; writers use std::atomic_store.
    std::shared_ptr<const WorkingState> activeWorkingState;
    std::shared_ptr<const DirectRenderState> activeDirectState;
    std::shared_ptr<const PrintRenderState> activePrintState;
    uint64_t activeBuildCounter = 0;
    std::atomic<std::uint64_t> buildCounterNext{0};
    std::atomic<std::uint32_t> frameBoundsVersion{0};
    OfxRectI cachedFrameBounds{0, 0, 0, 0};

    ParamSnapshot lastParams;
    std::atomic<std::uint64_t> lastHash{0};

    // Latest parameter snapshot observed from UI callbacks; consumed/coalesced on render thread.
    PendingParamsState pending;

    Print::Runtime printRT;

    bool suppressParamEvents = false;
    bool inBootstrap = false;

    std::string dataDir;
    bool baseLoaded = false;
    std::uint64_t sessionSeed = 0;
    std::uint64_t instanceToken = 0;
    std::atomic<std::uint64_t> submissionSnapshotIdNext{1};

    IlluminantOverrideFlags illuminantOverride;

    std::string filmReferenceIlluminant;

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    // Snapshot latch: all submissions for the same frame token reuse one immutable snapshot payload.
    std::mutex submissionSnapshotLatchMutex;
    bool submissionSnapshotLatchValid = false;
    JuicerCuda::ResourceManager::SubmissionSnapshot submissionSnapshotLatch{};
#endif
};

bool spektrafilm_profile_catalog_ready();
const char* spektrafilm_profile_catalog_failure();
int film_profile_option_count();
const char* film_profile_option_key(int index);
const char* film_profile_option_label(int index);
int print_profile_option_count();
const char* print_profile_option_key(int index);
const char* print_profile_option_label(int index);
bool load_film_profile_into_base(const std::string& filmProfileKey, InstanceState& S);
bool load_selected_spektrafilm_film_profile_into_base(
    const Profiles::ValidatedFilmProfile& profile,
    InstanceState& S);
bool rebuild_print_render_state(InstanceState& S, const ParamSnapshot& P);
void rebuild_working_state(OfxImageEffectHandle instance, InstanceState& S, const ParamSnapshot& P);
void rebuild_working_state_couplers_only(OfxImageEffectHandle instance, InstanceState& S, const ParamSnapshot& P);
