#pragma once

#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
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

namespace OFX {
    class Image;
}

namespace ScannerOptics {

    struct PlaneView {
        float* r = nullptr;
        float* g = nullptr;
        float* b = nullptr;
        float* a = nullptr;
        int width = 0;
        int height = 0;
        std::ptrdiff_t strideBytes = 0;
        int originX = 0;
        int originY = 0;
    };

    struct GlareCache {
        // Single glare scalar per pixel, blurred once and shared across XYZ.
        std::vector<float> amount;
        std::vector<float> tmp;
        Scanner::ScannerKey key{};
        std::uint64_t seedHash = 0;
        int width = 0;
        int height = 0;
        bool valid = false;
    };

    struct Runtime {
        Scanner::SpectralLutBuffer lut;
        std::vector<double> blurKernel;
        std::vector<double> unsharpKernel;
        float unsharpAmount = 0.0f;
        std::vector<double> rgbR;
        std::vector<double> rgbG;
        std::vector<double> rgbB;
        std::vector<double> scratchTmp;
        std::vector<double> scratchBlurred;
        GlareCache glare;
        Scanner::ScannerKey key{};
    };

    struct RenderAbortHandle {
        std::function<bool()> shouldAbort;
        bool abortRequested() const {
            return shouldAbort && shouldAbort();
        }
    };

    struct RenderContext {
        const Scanner::ScannerMediumRuntime* medium = nullptr;
        const Scanner::ScannerDensityBuffer* density = nullptr;
        Runtime* runtime = nullptr;
        OFX::Image* srcImage = nullptr;
        const Scanner::ColorRuntime* color = nullptr;
        int nComponents = 0;
        OfxRectI bounds{};
        PlaneView dstView{};
        Scanner::Options options{};
        Scanner::Settings settings{};
        Scanner::ScannerRuntimeKey runtimeKey{};
        Scanner::ScannerKey scannerKey{};
        std::uint64_t seedBase = 0;
        bool hasBaseline = false;
        unsigned int threadCount = 1;
        bool copyAlpha = false;
        RenderAbortHandle abort{};
    };

    Scanner::ColorRuntime build_color_runtime(
        const Scanner::ScannerMediumRuntime& medium,
        const OutputEncoding::Params& outputEncoding);

    void render_density_to_rgb(const RenderContext& ctx);

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
    Spectral::Curve baseMin, baseMid;
    std::array<std::array<std::vector<float>, 3>, 3> densityCurvesLayers{}; // [layer][channel] values on LOG_EXPOSURE axis
    bool hasDensityCurvesLayers = false;
    float dyeDensityMinFactor = 1.0f;
    std::array<float, 3> gammaFactor{{1.0f, 1.0f, 1.0f}};
    bool hasBaseline = false;
    std::string referenceIlluminant;
    std::string viewingIlluminant;
    std::vector<float> densityMidNeutral;
    std::vector<float> logExposureMidNeutral;
    Profiles::DirCouplersProfile dirCouplers;
    Profiles::MaskingCouplersProfile maskingCouplers;
    std::array<float, 3> cameraFilterUV{{1.0f, 410.0f, 8.0f}};
    std::array<float, 3> cameraFilterIR{{1.0f, 675.0f, 15.0f}};
    bool cameraFilterDefined = false;
    Profiles::GrainMetadata grain;
    Profiles::HalationMetadata halation;
    Profiles::ProfileGlare glare;
};

namespace WorkingStateSharing {
    struct WorkingStateCoreShared;
}

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

    Spectral::Curve baseMin;
    Spectral::Curve baseMid;
    bool hasBaseline = false;
    float baselineMixReference = 0.0f;
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
    Scanner::ColorRuntime negativeColorRuntime;
    Scanner::ScannerMediumRuntime negativeMediumRuntime;

    Scanner::ScannerIlluminant printScannerIlluminant;
    Scanner::ScannerDensityRange printDensityRange;
    Scanner::ScannerStaticKey printStaticKey;
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
    std::shared_ptr<const WorkingStateSharing::WorkingStateCoreShared> sharedCore;
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

    const Spectral::Curve& precorrectedB = ws.dirPrecorrected ? ws.dirDensB : ws.densB;
    const Spectral::Curve& precorrectedG = ws.dirPrecorrected ? ws.dirDensG : ws.densG;
    const Spectral::Curve& precorrectedR = ws.dirPrecorrected ? ws.dirDensR : ws.densR;

#ifdef JUICER_ENABLE_COUPLERS
    if (mode == DirSampleMode::ApplyRuntime && dirRT.active) {
        float layerPre[3];
        sample_layers(ws.densB, ws.densG, ws.densR, logE, layerPre);
        Couplers::ApplyInputLogE io{{logE[0], logE[1], logE[2]}, {layerPre[0], layerPre[1], layerPre[2]}};
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

struct ParamSnapshot {
    std::string filmProfileKey = Spektrafilm::kDefaultFilmProfileKey;
    std::string printProfileKey = Spektrafilm::kDefaultPrintProfileKey;
    Spektrafilm::ScanRoute scanRoute = Spektrafilm::kDefaultScanRoute;
    std::uint64_t filmProfileAssetVersionToken = 0;
    std::uint64_t printProfileAssetVersionToken = 0;
    bool directRoutePrintProfileExcluded = false;
    bool directRouteNeutralCalibrationExcluded = false;
    int spectralUpsamplingMode = 0;
    int refIll = 0;
    int enlIll = 3;
    int enlDichroicSet = 0;
    double printExposure = 1.0;
    double printPreflashExposure = 0.0;
    int normalizePrintExposure = 1;
    int printExposureCompensation = 1;
    std::array<double, 3> printUiYmcCc{};
    double preflashMFilterCc = 0.0;
    double preflashYFilterCc = 0.0;
    double glareCompRemovalFactor = 0.0;
    double glareCompRemovalDensity = 1.2;
    double glareCompRemovalTransition = 0.3;
    bool glareActive = true;
    double glarePercent = 0.10;
    double glareRoughness = 0.4;
    double glareBlurSigmaPx = 0.5;
    double printDminFactor = 0.4;
    int couplersActive = 1;
    double couplersAmount = 1.0;
    double ratioR = 1.0, ratioG = 1.0, ratioB = 1.0;
    double sigma = 2.0, high = 0.0;
    int inputColorSpace = Spectral::inputColorSpaceToIndex(Spectral::InputColorSpace::DaVinciWideGamut);
    int inputCctfDecoding = 0;
    int cameraAutoExposureEnabled = 1;
    int cameraMeteringMethod = 0;
    double cameraExposureCompensationEv = 0.0;
    double cameraFilmFormatLongEdgeMm = 36.0;
    double scannerLensBlurSigmaPx = 0.55;
    std::array<double, 2> scannerUnsharpMask{{0.7, 1.0}};
    int scannerUseLut = 1;
    int scannerLutResolution = 17;
    double spatialSigmaMicrometers = 10.0;
    int outputColorSpace = OutputEncoding::toIndex(OutputEncoding::ColorSpace::sRGB);
    int outputCctfEncoding = 1;
    int outputLinearPassThrough = 0;
    bool cameraFilterOverride = false;
    std::array<double, 3> cameraFilterUV{{1.0, 410.0, 8.0}};
    std::array<double, 3> cameraFilterIR{{1.0, 675.0, 15.0}};
};

constexpr int kFactoryCouplersActive = 1;
constexpr double kFactoryCouplersAmount = 1.0;
constexpr double kFactoryCouplersRatioR = 1.0;
constexpr double kFactoryCouplersRatioG = 1.0;
constexpr double kFactoryCouplersRatioB = 1.0;
constexpr double kFactoryCouplersSigma = 2.0;
constexpr double kFactoryCouplersHigh = 0.0;
constexpr double kFactoryCouplersSpatialSigma = 10.0;

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

    bool couplerProfileSpatialSigmaValid = false;
    double couplerProfileSpatialSigmaMicrometers = 0.0;

    // Cache for DIR spatial sigma conversion (canonical project dimensions)
    std::atomic<bool> spatialSigmaCacheValid{false};
    std::atomic<double> spatialSigmaCanonicalWidth{0.0};
    std::atomic<double> spatialSigmaCanonicalHeight{0.0};
    std::atomic<double> spatialSigmaCameraFilmMm{0.0};
    std::atomic<float> spatialSigmaMicrometers{0.0f};
    std::atomic<float> spatialSigmaPixelsCanonical{0.0f};

    std::string filmReferenceIlluminant;

    // Auto-exposure cache (per frame / build)
    std::mutex autoExposureMutex;
    bool autoExposureCacheValid = false;
    bool autoExposureCacheIsCudaRender = false;
    double autoExposureCacheTime = std::numeric_limits<double>::quiet_NaN();
    bool autoExposureCacheAutoEnabled = false; // Tracks camera auto-exposure toggle state
    int autoExposureCacheMeteringMethod = 0;
    uint64_t autoExposureCacheBuildCounter = 0;
    OfxRectI autoExposureCacheBounds{0, 0, 0, 0};
    double autoExposureCacheEV = 0.0;
    double autoExposureCacheRenderScaleX = 0.0;
    double autoExposureCacheRenderScaleY = 0.0;
    std::uintptr_t autoExposureCacheClipToken = 0;
    int autoExposureCacheInputColorSpaceIndex =
        Spectral::inputColorSpaceToIndex(Spectral::InputColorSpace::DaVinciWideGamut);
    bool autoExposureCacheApplyCctfDecoding = false;
    bool autoExposureMaskValid = false;
    int autoExposureMaskWidth = 0;
    int autoExposureMaskHeight = 0;
    double autoExposureMaskSigma = 0.0;
    double autoExposureMaskSum = 0.0;
    double autoExposureMaskRenderScaleX = 0.0;
    double autoExposureMaskRenderScaleY = 0.0;
    std::uintptr_t autoExposureMaskClipToken = 0;
    std::shared_ptr<const std::vector<double>> autoExposureMaskWeights;
    std::uint64_t autoExposureMaskCachedBytes = 0;
    bool autoExposureMaskPolicyBypass = false;
    std::uint64_t autoExposureMaskLastRequestedBytes = 0;

    ScannerOptics::Runtime scannerRuntimeA;
    ScannerOptics::Runtime scannerRuntimeB;
    std::atomic<bool> scannerRuntimeAInUse{false};
    std::atomic<bool> scannerRuntimeBInUse{false};

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
