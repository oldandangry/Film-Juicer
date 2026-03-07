#pragma once

#if defined(_MSC_VER)
// MSVC STL warns (and this project treats warnings as errors) about shared_ptr atomic free-functions
// being deprecated in C++20. We intentionally use them for C++17 compatibility.
#ifndef _SILENCE_CXX20_OLD_SHARED_PTR_ATOMIC_SUPPORT_DEPRECATION_WARNING
#define _SILENCE_CXX20_OLD_SHARED_PTR_ATOMIC_SUPPORT_DEPRECATION_WARNING
#endif
#endif

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

#include "Print.h"
#include "ProfileJSONLoader.h"
#include "ColorTransforms.h"
#include "SpectralData.h"
#include "ofxImageEffect.h"
#include "ScannerOptics.h"

extern const std::string gDataDir;
struct WorkingState;

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
#include "Cuda/ResourceManager/JuicerCudaResourceTypes.h"

namespace JuicerCuda {
    struct Resources;
    void destroy(Resources* resources) noexcept;
}

struct JuicerCudaResourcesDeleter {
    void operator()(JuicerCuda::Resources* resources) const noexcept;
};
#endif

inline std::filesystem::path data_dir_path() {
    std::filesystem::path path(gDataDir);
    path.make_preferred();
    return path;
}

template <typename... Parts>
inline std::filesystem::path data_dir_path(Parts&&... parts) {
    std::filesystem::path path(gDataDir);
    (void)std::initializer_list<int>{
        ((path /= std::filesystem::path(std::forward<Parts>(parts))), 0)...
    };
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
    std::array<float, 3> gammaFactor{ {1.0f, 1.0f, 1.0f} };
    bool hasBaseline = false;
    std::string referenceIlluminant;
    std::string viewingIlluminant;
    std::vector<float> densityMidNeutral;
    std::vector<float> logExposureMidNeutral;
    Profiles::DirCouplersProfile dirCouplers;
    Profiles::MaskingCouplersProfile maskingCouplers;
    std::array<float, 3> cameraFilterUV{ {1.0f, 410.0f, 8.0f} };
    std::array<float, 3> cameraFilterIR{ {1.0f, 675.0f, 15.0f} };
    bool cameraFilterDefined = false;
    Profiles::GrainMetadata grain;
    Profiles::HalationMetadata halation;
    Profiles::ProfileGlare glare;
};

struct ParamSnapshot {
    int filmStockIndex = 0;
    int printPaperIndex = 0;
    int spectralUpsamplingMode = 0;
    int refIll = 0;
    int enlIll = 3;
    int enlDichroicSet = 0;
    double glareCompRemovalFactor = 0.0;
    double glareCompRemovalDensity = 1.2;
    double glareCompRemovalTransition = 0.3;
    double printDminFactor = 0.4;
    int couplersActive = 1;
    double couplersAmount = 1.0;
    double ratioR = 1.0, ratioG = 1.0, ratioB = 1.0;
    double sigma = 2.0, high = 0.0;
    int inputColorSpace = Spectral::inputColorSpaceToIndex(Spectral::InputColorSpace::DaVinciWideGamut);
    int inputCctfDecoding = 0;
    double scannerLensBlurSigmaPx = 0.55;
    std::array<double, 2> scannerUnsharpMask{ {0.7, 1.0} };
    int scannerUseLut = 1;
    int scannerLutResolution = 17;
    double spatialSigmaMicrometers = 10.0;
    int outputColorSpace = OutputEncoding::toIndex(OutputEncoding::ColorSpace::sRGB);
    int outputCctfEncoding = 1;
    int outputLinearPassThrough = 0;
    bool cameraFilterOverride = false;
    std::array<double, 3> cameraFilterUV{ {1.0, 410.0, 8.0} };
    std::array<double, 3> cameraFilterIR{ {1.0, 675.0, 15.0} };
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

struct CouplerDirtyFlags {
    std::atomic<bool> active{ false };
    std::atomic<bool> amount{ false };
    std::atomic<bool> ratioR{ false };
    std::atomic<bool> ratioG{ false };
    std::atomic<bool> ratioB{ false };
    std::atomic<bool> sigma{ false };
    std::atomic<bool> high{ false };
    std::atomic<bool> spatialSigma{ false };
};

struct IlluminantOverrideFlags {
    bool reference = false;
    bool enlarger = false;
};

struct PendingParamsState {
    std::mutex m;
    ParamSnapshot params;
    std::uint64_t fullHash = 0;
    std::uint64_t coreHash = 0;
    std::uint64_t dirHash = 0;
};

struct InstanceState {
    std::mutex m;
    BaseState base;
    // Published render snapshot. Readers use std::atomic_load; writers use std::atomic_store.
    std::shared_ptr<const WorkingState> activeWorkingState;
    uint64_t activeBuildCounter = 0;
    std::atomic<std::uint64_t> buildCounterNext{ 0 };
    std::atomic<std::uint32_t> frameBoundsVersion{ 0 };
    OfxRectI cachedFrameBounds{ 0, 0, 0, 0 };

    ParamSnapshot lastParams;
    std::atomic<std::uint64_t> lastHash{ 0 };

    // Latest parameter snapshot observed from UI callbacks; consumed/coalesced on render thread.
    PendingParamsState pending;

    Print::Runtime printRT;

    bool suppressParamEvents = false;
    bool inBootstrap = false;

    std::string dataDir;
    bool baseLoaded = false;
    std::uint64_t sessionSeed = 0;
    std::uint64_t instanceToken = 0;
    std::atomic<std::uint64_t> submissionSnapshotIdNext{ 1 };

    CouplerDirtyFlags couplerDirty;
    IlluminantOverrideFlags illuminantOverride;

    bool couplerProfileSpatialSigmaValid = false;
    double couplerProfileSpatialSigmaMicrometers = 0.0;

    // Cache for DIR spatial sigma conversion (canonical project dimensions)
    std::atomic<bool> spatialSigmaCacheValid{ false };
    std::atomic<double> spatialSigmaCanonicalWidth{ 0.0 };
    std::atomic<double> spatialSigmaCanonicalHeight{ 0.0 };
    std::atomic<double> spatialSigmaCameraFilmMm{ 0.0 };
    std::atomic<float> spatialSigmaMicrometers{ 0.0f };
    std::atomic<float> spatialSigmaPixelsCanonical{ 0.0f };

    std::string filmReferenceIlluminant;

    // Auto-exposure cache (per frame / build)
    std::mutex autoExposureMutex;
    bool autoExposureCacheValid = false;
    bool autoExposureCacheIsCudaRender = false;
    double autoExposureCacheTime = std::numeric_limits<double>::quiet_NaN();
    bool autoExposureCacheAutoEnabled = false; // Tracks camera auto-exposure toggle state
    int autoExposureCacheMeteringMethod = 0;
    uint64_t autoExposureCacheBuildCounter = 0;
    OfxRectI autoExposureCacheBounds{ 0, 0, 0, 0 };
    double autoExposureCacheEV = 0.0;
    double autoExposureCacheRenderScaleX = 0.0;
    double autoExposureCacheRenderScaleY = 0.0;
    std::uintptr_t autoExposureCacheClipToken = 0;
    int autoExposureCacheInputColorSpaceIndex =
        Spectral::inputColorSpaceToIndex(Spectral::InputColorSpace::DaVinciWideGamut);
    bool autoExposureCacheApplyCctfDecoding = false;
    bool autoExposureCanonicalValid = false;
    OfxRectI autoExposureCanonicalBounds{ 0, 0, 0, 0 };
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

    // Print mid-gray (kMidSpectral) cache: avoids recomputing the mid-gray probe for every render call.
    // Keyed by WorkingState.buildCounter and the small set of print parameters that affect the probe.
    std::mutex printMidgrayMutex;
    bool printMidgrayValid = false;
    std::uint64_t printMidgrayBuildCounter = 0;
    float printMidgrayYShiftSteps = 0.0f;
    float printMidgrayMShiftSteps = 0.0f;
    float printMidgrayCShiftSteps = 0.0f;
    float printMidgrayExposureCompScale = 1.0f;
    std::uint64_t printMidgrayNeutralFilterHash = Print::kDefaultNeutralFilterHash;
    float printMidgrayFactor = 1.0f;

    ScannerOptics::Runtime scannerRuntimeA;
    ScannerOptics::Runtime scannerRuntimeB;
    std::atomic<bool> scannerRuntimeAInUse{ false };
    std::atomic<bool> scannerRuntimeBInUse{ false };

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    // CUDA: per-instance GPU cache keyed by WorkingState.buildCounter.
    // This is kept on InstanceState so the CPU and CUDA render paths share the same invalidation
    // boundary (the atomic WorkingState swap).
    std::mutex cudaMutex;
    std::unordered_map<
        JuicerCuda::ResourceManager::DeviceContextKey,
        std::unique_ptr<JuicerCuda::Resources, JuicerCudaResourcesDeleter>,
        JuicerCuda::ResourceManager::DeviceContextKeyHash> cudaByDevice;

    // Snapshot latch: all submissions for the same frame token reuse one immutable snapshot payload.
    std::mutex submissionSnapshotLatchMutex;
    bool submissionSnapshotLatchValid = false;
    JuicerCuda::ResourceManager::SubmissionSnapshot submissionSnapshotLatch{};
#endif
};

std::string print_dir_for_index(int index);
const char* print_paper_json_key_for_index(int index);
const char* negative_json_key_for_stock_index(int filmIndex);
int film_stock_option_count();
const char* film_stock_option_label(int index);
int print_paper_option_count();
const char* print_paper_option_label(int index);
bool load_film_stock_into_base(int filmIndex, InstanceState& S);
void rebuild_working_state(OfxImageEffectHandle instance, InstanceState& S, const ParamSnapshot& P);
void rebuild_working_state_couplers_only(OfxImageEffectHandle instance, InstanceState& S, const ParamSnapshot& P);
